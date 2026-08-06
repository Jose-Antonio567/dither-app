# Gerando a build Windows

No PowerShell, na raiz do projeto:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
.\build.ps1
```

O executável estará em `dist\DitherApp\DitherApp.exe`. Esta primeira versão é
**onedir**: distribua toda a pasta `dist\DitherApp`, não somente o EXE. Onefile
gera um arquivo único, mas terá inicialização diferente e não é usado nesta etapa.
Gere a build Windows no próprio Windows.
