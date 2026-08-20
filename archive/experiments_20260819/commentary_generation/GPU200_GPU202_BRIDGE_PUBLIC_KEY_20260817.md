# GPU200 dedicated key authorization

Local private key:

`/home/tianlin/.ssh/id_ed25519_gpu200_gpu202_bridge`

Public-key fingerprint:

`SHA256:T5sFuy2R8FOj4sG0DueQ9U4/4xlXb69fXWrDqOlMYcU`

The private key has no passphrase so the diagnostic transfer can run
noninteractively. It must remain only on gpu200 with mode 600. Do not copy it
to the repository, NAS, chat, or another host.

## Add on bastion-DB13

Add the following single line to the `jumpserver` account's
`~/.ssh/authorized_keys`. It disables agent/X11/PTY use and permits TCP
forwarding only to the intended GPU202 SSH endpoint.

```text
no-agent-forwarding,no-X11-forwarding,no-pty,permitopen="172.16.11.202:15654" ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBRO7PUAU92ARl30WSuXUUASBz50l80ZeP4zvKyydQRF tianlin@gpu200 dedicated GPU202 route 2026-08-17
```

## Add on GPU202-tianlin

Add the following single line to the `tianlin` account's
`~/.ssh/authorized_keys`. `restrict` disables PTY, forwarding, agent
forwarding, X11 forwarding and user rc while still allowing required remote
commands such as `stat`, `du` and `rsync`.

```text
restrict ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBRO7PUAU92ARl30WSuXUUASBz50l80ZeP4zvKyydQRF tianlin@gpu200 dedicated GPU202 route 2026-08-17
```

Do not remove or replace existing authorized keys. Before appending, check
whether the fingerprint or key body is already present. Ensure the remote
`.ssh` directory is mode 700 and `authorized_keys` is mode 600.

After both entries are installed, compare the bastion ED25519 fingerprint
observed from gpu200 with the Mac's already-trusted record. Only after a match
should the host key be added locally and the batch-mode login tested.
