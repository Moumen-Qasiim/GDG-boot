# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['/home/momn/Projects/GDG-boot/tic-tac-toe-cv/main.py'],
    pathex=[],
    binaries=[],
    datas=[('/home/momn/Projects/GDG-boot/tic-tac-toe-cv/hand_landmarker.task', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Tic_Tac_Toe_CV',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
