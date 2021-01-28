# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

project_files=[
    ( 'examples/*', '.' ),
    ( 'assets/*', '.' )
]


a = Analysis(['src/_run_main.py', '_run_main.spec'],
             pathex=['/Users/kevin/Documents/Repos/TeachApp'],
             binaries=[("/System/Library/Frameworks/Foundation.framework/Versions/C/Resources/BridgeSupport/Foundation.dylib",".")],
             datas=project_files,
             hiddenimports=["cmath","sklearn.neighbors._typedefs","sklearn.utils._weight_vector","sklearn.neighbors._quad_tree"],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='main',
          debug=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=False)

app = BUNDLE(exe,
         name='TeachApp.app',
         icon=None,
         bundle_identifier="com.chevalier.teachapp.zip",
)