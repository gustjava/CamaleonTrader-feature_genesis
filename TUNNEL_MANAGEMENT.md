# Gerenciamento de Túneis SSH Persistente

## Problema Resolvido

Anteriormente, o túnel SSH para conectar ao MySQL local caía quando a sessão de trabalho era encerrada. Agora o túnel é criado com `nohup` e roda em background, persistindo mesmo após fechar a sessão.

## Como Funciona

### 1. Criação do Túnel Persistente

O script `deploy_to_vast.sh` agora:

- Cria um túnel SSH em background usando `nohup`
- Salva o PID do processo em `/tmp/vast_tunnel_<INSTANCE_ID>.pid`
- Salva os logs em `/tmp/vast_tunnel_<INSTANCE_ID>.log`
- Verifica se o túnel está funcionando antes de prosseguir

### 2. Gerenciamento de Túneis

Use o script `manage_tunnels.sh` para gerenciar os túneis:

```bash
# Listar todos os túneis ativos
./manage_tunnels.sh list

# Parar túnel de uma instância específica
./manage_tunnels.sh stop <INSTANCE_ID>

# Parar todos os túneis
./manage_tunnels.sh stop-all
```

## Arquivos Criados

Para cada instância, são criados:

- `/tmp/vast_tunnel_<INSTANCE_ID>.pid` - PID do processo do túnel
- `/tmp/vast_tunnel_<INSTANCE_ID>.log` - Logs do túnel SSH

## Comandos Úteis

```bash
# Verificar se túnel está ativo
ps aux | grep 'ssh.*<SSH_HOST>'

# Ver logs em tempo real
tail -f /tmp/vast_tunnel_<INSTANCE_ID>.log

# Parar túnel manualmente
kill $(cat /tmp/vast_tunnel_<INSTANCE_ID>.pid)

# Verificar se porta está aberta
nc -z -w5 127.0.0.1 3010
```

## Vantagens

1. **Persistência**: Túnel continua rodando após fechar a sessão
2. **Gerenciamento**: Scripts para listar e parar túneis facilmente
3. **Logs**: Arquivos de log para debug
4. **Limpeza**: Remove túneis órfãos automaticamente
5. **Verificação**: Testa se o túnel está funcionando antes de usar

## Fluxo de Trabalho

1. Execute `./deploy_to_vast.sh`
2. O script cria o túnel persistente automaticamente
3. Execute seu pipeline normalmente
4. Feche a sessão - o túnel continua rodando
5. Use `./manage_tunnels.sh list` para verificar túneis ativos
6. Use `./manage_tunnels.sh stop <ID>` quando não precisar mais
