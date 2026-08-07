# Instalador Windows

O instalador usa a build **onedir** em `dist/DitherApp`, porque ela inclui todas
as dependências Qt necessárias. Instale o Inno Setup 6 ou 7 e execute:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\build-installer.ps1
```

O arquivo será criado em `release/installer`. Teste instalação, execução e
desinstalação em uma conta de usuário comum. O AppId não deve ser alterado: ele
permite atualizar instalações futuras sem duplicar a entrada do aplicativo.
Atualize os valores de versão no script `.iss` nas próximas releases. Esta build
ainda não possui assinatura digital.
