# 实验室集群管理员学习包

更新日期：2026-08-14  
案例主机：`gpu200`  
主要主题：Linux、网络、共享存储、IBM Storage Scale（GPFS）和故障排查

## 1. 如何使用这份文档

这是一份面向实验室环境的实用学习材料，不采用繁重的企业流程。可以把整个文件上传或粘贴给网页版 ChatGPT，让它：

1. 按章节讲解概念；
2. 根据当前 GPU200 故障出练习题；
3. 逐行解释终端输出；
4. 区分只读检查和会改变集群状态的操作；
5. 在没有足够证据时明确说“未知”，不要猜测。

文档不包含密码、令牌、SSH 私钥或其他登录凭证。

## 2. 学习目标

最终应能独立完成以下工作：

- 看懂一台 GPU 节点的 CPU、内存、GPU、磁盘、网络和挂载状态；
- 区分本地磁盘、NAS、NFS/SSHFS 和 GPFS；
- 画出计算节点、管理网络、存储网络、quorum 节点和数据服务器的关系；
- 判断问题发生在程序、挂载、GPFS 客户端、网络、quorum 还是数据服务器；
- 先收集证据，再选择最小修复；
- 知道哪些操作只影响一台客户端，哪些可能影响整个集群；
- 故障恢复后安全地恢复训练，而不是直接覆盖 checkpoint。

## 3. 先理解整个访问链路

正常情况下，程序读取 `/remote-home` 的链路大致如下：

```text
Python / 训练程序
        │
        ▼
Linux 路径 /remote-home
        │
        ▼
GPU 节点上的 GPFS 客户端 mmfsd
        │
        ▼
GPU 节点的存储网卡和存储网络
        │
        ├── quorum / cluster manager / CCR
        │       确认节点身份并维护集群一致性
        │
        └── NSD / 数据服务器 / 磁盘
                真正提供文件数据
```

quorum 节点不一定保存所有文件内容。它们首先负责确保集群仍然具备合法多数，防止网络分区后不同节点同时修改同一份数据。

## 4. 常用名词

### 本地文件系统

数据直接放在当前服务器的硬盘上。例如 GPU200 的 `/home/tianlin` 位于本地 `ext4` 文件系统。其他机器通常看不到这份数据。

### NAS

独立的网络存储设备。服务器通过网络协议访问它。`/mnt/nas`、`/mnt/nas2` 属于此类挂载，但具体协议和性能需要分别确认。

### GPFS / IBM Storage Scale

GPFS 是 IBM Storage Scale 的旧称。它是一种面向集群的并行共享文件系统，可以让多台 GPU 和存储节点共同访问同一套文件。

### GPFS client

安装并运行 GPFS 客户端的计算节点，例如 `gpu200`。客户端把共享文件系统挂载成普通目录供程序使用。

### quorum

集群的“法定多数”。节点必须能联系足够数量的 quorum 节点，才能继续访问文件系统。失去 quorum 时停止访问，是为了保护数据一致性。

### CCR

Cluster Configuration Repository，用于维护和同步集群配置。大量 GPFS 管理命令依赖 CCR 可用。

### lease

节点定期续签的集群成员凭证。lease 过期通常说明节点长时间无法正常联系集群管理节点。

### NSD

Network Shared Disk。它描述 GPFS 如何通过服务器向集群节点提供底层磁盘。

### Stale file handle

Linux 还保留着旧的挂载或文件句柄，但背后的远程文件系统已经断开或失效。挂载表中仍显示目录，不代表数据真的可以读取。

## 5. GPU200 本次真实案例

### 已确认事实

- 主机：`gpu200`；用户：`tianlin`；UID 1029，GID 1031。
- `/remote-home` 的文件系统来源显示为 `gpfsdata`，类型为 `gpfs`。
- 挂载记录仍存在，但访问 `/remote-home` 返回：

  ```text
  Stale file handle
  ```

- GPU200 的 GPFS 服务状态为：

  ```text
  inactive
  ```

- GPFS 自身报告：

  ```text
  GPFS is down on this node.
  ```

- 集群配置记录了四个节点：

  ```text
  gpfs204  172.16.10.204  quorum-manager
  gpfs205  172.16.10.205  quorum-manager
  gpu200   172.16.10.200
  gpu202   172.16.10.202
  ```

- GPU200 到 `172.16.10.204` 和 `172.16.10.205` 的测试均为 100% 丢包。
- GPU200 当前没有 `172.16.10.200` 地址。
- 当前正常工作的 `eno3` 使用 `172.16.11.200/24`；访问 `172.16.10.204/205` 时流量会经 `172.16.11.254` 绕行。
- 多个疑似高速或存储网络接口处于 `DOWN`，其中一个接口显示 `NO-CARRIER`。
- `gpu202` 主机名在 GPU200 上解析为 `172.16.10.202`，因此从 GPU200 进行 SSH 对比也会超时；这不能证明 GPU202 本身宕机。

### 日志时间线

```text
2026-08-14 04:12:35  Disk lease period expired
2026-08-14 04:13:02  Bad TCP state detected
                     Connection reset by peer
2026-08-14 04:13:04  无法从 gpfs204 获得 quorum 响应
2026-08-14 04:13:14  无法从 gpfs205 获得 quorum 响应
2026-08-14 04:13:14  Unable to contact any quorum nodes
2026-08-14 04:13:14  Lost membership in cluster
                     Unmounting file systems
```

### 当前最可能的故障点

```text
GPU200 上原本承载 172.16.10.200 的接口或配置
        │
        ├── 网卡/驱动/系统网络配置
        ├── 物理链路或光模块
        ├── 交换机端口
        └── VLAN 或存储网络路由
```

GPU200 缺少 `172.16.10.200` 已足以解释为什么它无法联系 quorum。但是目前仍不能排除 `gpfs204/gpfs205` 同时存在服务端故障。

## 6. 标准排查顺序

不要一看到文件系统故障就直接重启。按下面顺序缩小范围：

```text
1. 应用层：只有一个程序失败，还是所有程序都失败？
2. 路径层：目录是否能 stat/读取？
3. 挂载层：挂载记录和真实访问是否一致？
4. 客户端层：本机 GPFS daemon 是否运行？
5. 网络层：正确的 IP、接口、链路和路由是否存在？
6. 集群层：quorum、CCR 和 membership 是否正常？
7. 数据层：NSD、磁盘和文件系统是否正常？
8. 恢复层：网络和集群健康后再启动客户端、挂载和恢复任务。
```

一个实用原则：先判断故障范围。

- 只有 GPU200 异常：优先检查 GPU200 的网卡、配置和交换机端口。
- GPU200、GPU202 都异常：优先检查共享网络或 quorum 节点。
- 网络恢复但 GPFS 仍异常：检查 quorum、CCR、membership 和文件系统状态。

## 7. 只读检查工具

下面的命令通常用于观察状态，但仍应根据账号权限和实验室规范执行。

### 主机与权限

```bash
hostname
id
uptime
```

### 文件系统与挂载

```bash
findmnt -T /remote-home
timeout 5s stat /remote-home
df -hT
lsblk
```

对失效的网络文件系统使用 `timeout`，避免命令长期卡住。

### 网络

```bash
ip -brief address
ip -brief link
ip route show
ip route get 172.16.10.204
ping -c 2 172.16.10.204
ping -c 2 172.16.10.205
ss -ltnp
```

`ping` 失败不一定代表服务端宕机，也可能是 ICMP 被屏蔽、路由错误或本机接口缺失，必须结合其他证据判断。

### GPFS

GPFS 工具一般位于 `/usr/lpp/mmfs/bin`：

```bash
/usr/lpp/mmfs/bin/mmlscluster
/usr/lpp/mmfs/bin/mmgetstate -a
/usr/lpp/mmfs/bin/mmlsmount all -L
/usr/lpp/mmfs/bin/mmlsfs all
/usr/lpp/mmfs/bin/mmlsnsd
/usr/lpp/mmfs/bin/mmlsdisk all
/usr/lpp/mmfs/bin/mmlsconfig
```

部分命令可能因本地权限不足而无法执行。这是权限信息，不应被误判为 GPFS 故障。

### 日志

常见客户端日志：

```text
/var/adm/ras/mmfs.log.latest
```

重点搜索：

```text
lease
quorum
CCR
TCP
membership
unmount
expel
```

## 8. 会改变状态的操作

以下操作不是普通检查，执行前至少要知道影响范围和回滚方式：

- 启动或停止 GPFS daemon；
- 挂载或卸载 GPFS 文件系统；
- 修改 IP、路由、网卡、bond、VLAN 或 MTU；
- 重启网卡、交换机端口、quorum 节点或 NSD 节点；
- 修改 quorum 角色、CCR 配置、NSD、磁盘或文件系统；
- 强制卸载、清理集群状态或同时重启多个 quorum 节点。

实验室可以不采用复杂审批，但至少应做到：

1. 操作前保存状态；
2. 一次只改变一个变量；
3. 明确影响一台机器还是整个集群；
4. 记录如何撤销；
5. 不同时重启所有 quorum 节点。

## 9. 权限应该如何理解

权限不是简单的“有没有 sudo”，而是分层的：

| 层级 | 典型能力 |
|---|---|
| 普通实验用户 | 查看自己的进程、使用 GPU、读写授权目录、启动自己的训练 |
| GPU 节点管理员 | 管理本机网络、服务、驱动和挂载 |
| 网络管理员 | 管理交换机端口、VLAN、路由和物理链路 |
| GPFS 管理员 | 管理 quorum、CCR、节点、NSD、磁盘和文件系统 |

拥有 GPU200 的 sudo，不一定能登录 `gpfs204/gpfs205`，也不一定具备 GPFS 集群管理权限。

当前 SoccerMaster 项目会话遵循以下边界：

- 不使用 sudo；
- 不修改共享原始仓库和共享 Python 环境；
- 不读取或传播密码、私钥和令牌；
- 集群恢复操作交给具备相应权限的人。

## 10. 建议学习路线

### 第 1 周：Linux 与网络

- 用户、权限、进程、systemd 和日志；
- 磁盘、分区、文件系统和挂载；
- IP、子网、路由、DNS、端口、VLAN；
- 网卡 `UP/DOWN`、`LOWER_UP` 和 `NO-CARRIER` 的区别。

练习目标：能够解释 GPU200 为什么还能 SSH 登录，却无法访问 GPFS。

### 第 2 周：共享存储

- 本地盘、RAID、NAS、NFS、SSHFS、GPFS；
- 容量、inode、quota；
- 网络文件系统失联和 stale handle；
- 共享存储性能由哪些部分决定。

练习目标：画出实验室真实数据从磁盘到训练程序的路径。

### 第 3 周：GPFS 基础

- client、manager、quorum、CCR、NSD；
- filesystem、fileset、storage pool；
- lease、membership 和故障恢复；
- 熟悉只读 `mmls*`、`mmgetstate`、`mmhealth` 和日志。

练习目标：只根据正常状态输出判断各节点角色。

### 第 4 周：故障演练

- 单客户端网络中断；
- quorum 节点不可达；
- GPFS daemon 停止；
- mount 表残留但文件不可访问；
- 磁盘空间、inode 或 quota 耗尽；
- 训练因共享存储中断退出后的 checkpoint 恢复。

故障演练应在测试环境、虚拟机或纸面推演中进行，不要主动破坏生产集群。

## 11. 本案例的下一步练习

从仍可登录 GPU202 的入口，在 GPU202 上做一次只读对比：

```bash
hostname
ip -brief address
ip route show
findmnt -T /remote-home
timeout 5s stat -c 'path=%n type=%F' /remote-home
systemctl is-active gpfs
/usr/lpp/mmfs/bin/mmlscluster
/usr/lpp/mmfs/bin/mmlsmount all -L
```

需要回答：

1. GPU202 是否拥有 `172.16.10.202`？
2. 它位于哪张网卡？
3. `/remote-home` 是否真的能读取？
4. GPFS daemon 是否运行？
5. 如果 GPU202 正常，GPU200 与它的关键差异是什么？

## 12. 官方学习资料

- [IBM Storage Scale Administrator 学习路径](https://www.ibm.com/training/learning-path/ibm-storage-scale-administrator-913)
- [IBM Storage Scale System Fundamentals](https://www.ibm.com/training/course/ibm-storage-scale-system-fundamentals-SSE13DG)
- [IBM 对 quorum nodes 的说明](https://www.ibm.com/docs/en/storage-scale-ece/5.2.3?topic=roles-quorum-nodes)
- [IBM Storage Scale 管理命令说明](https://www.ibm.com/docs/en/storage-scale/6.0.0?topic=scale-storage-administration-commands)
- [IBM mmstartup 命令说明](https://www.ibm.com/docs/en/storage-scale/6.0.0?topic=reference-mmstartup-command)

使用官方文档时，应优先选择与实验室实际安装版本一致的文档，不要直接把其他版本的命令复制到生产集群。

## 13. 可直接交给 ChatGPT 的提示词

```text
你是我的实验室 GPU 集群管理员导师。我刚开始学习 Linux、网络、共享存储和 IBM Storage Scale（GPFS）。

请完整阅读我上传的《实验室集群管理员学习包》，并遵守以下要求：

1. 使用通俗中文，不假设我已经懂存储术语。
2. 先解释原理，再解释命令，不要让我机械背命令。
3. 每次只讲一个小主题，然后给我一道基于 GPU200 真实故障的练习题。
4. 每条命令明确标记：只读检查、本机状态修改、网络修改或集群级修改。
5. 没有足够证据时明确标记“未知”，不要编造服务器拓扑。
6. 不要求我提供密码、私钥、令牌或其他秘密。
7. 不建议我在生产集群上人为制造故障。
8. 当我贴出终端输出时，请逐行解释，并指出正常值、异常值和下一条最小检查。
9. 学习顺序为：Linux基础 → 网络 → 文件系统与共享存储 → GPFS架构 → 故障排查 → 安全恢复。
10. 现在先从“为什么 GPU200 还能登录，但 /remote-home 已经不可访问”开始讲解。
```
