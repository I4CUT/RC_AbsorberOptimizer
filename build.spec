# build.spec — конфигурация сборки RC Absorber Optimiser в исполняемый файл.
#
# Использование (в активированном venv, в папке проекта):
#     pyinstaller build.spec
#
# Пересборка после изменений в коде — та же команда, ничего не нужно
# держать в голове или переписывать в командной строке.
#
# Результат появится в dist/RC_Absorber_Optimiser/RC_Absorber_Optimiser.exe

from PyInstaller.utils.hooks import collect_all

block_cipher = None

# ── Библиотеки, которые PyInstaller не всегда находит автоматически ────────
# (динамическая загрузка подмодулей — matplotlib backends, sklearn estimators,
# openpyxl reader/writer plugins)
datas = []
binaries = []
hiddenimports = [
    'sklearn.ensemble',
    'sklearn.ensemble._forest',
    'sklearn.tree',
    'sklearn.tree._tree',
    'sklearn.neighbors._typedefs',
    'sklearn.utils._typedefs',
    'sklearn.utils._heap',
    'sklearn.utils._sorting',
    'sklearn.utils._vector_sentinel',
]

for pkg in ('matplotlib', 'sklearn', 'openpyxl'):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RC_Absorber_Optimiser',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,   # без консольного окна; поставьте True для отладки
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='icon.ico',   # раскомментируйте, если есть файл иконки .ico
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='RC_Absorber_Optimiser',
)
