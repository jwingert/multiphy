# Build both entry points against one analysis and one shared dependency folder.
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

root = Path(SPECPATH).resolve().parents[1]
assets = root / 'build' / 'desktop' / 'assets'
excluded = [
    'torch', 'torchvision', 'torchaudio', 'tensorflow', 'jax', 'jaxlib',
    'UnitMatchPy', 'unitmatch', 'DANT', 'dant', 'deepunitmatch',
    'pytest', '_pytest', 'IPython', 'ipykernel', 'notebook', 'jupyter',
    'setuptools', 'pip', 'wheel', 'tkinter', 'PyQt5', 'PyQt6', 'PySide2',
    'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets',
    'PySide6.QtWebEngineQuick', 'PySide6.QtPdf', 'PySide6.QtPdfWidgets',
    'PySide6.QtVirtualKeyboard',
    'numpy.tests', 'pandas.tests', 'scipy.tests', 'matplotlib.tests',
]
datas = collect_data_files('multiphy', includes=['*.md', 'docs/*.md',
    'viewer/fonts/*.ttf', 'viewer/fonts/*.txt', 'viewer/logo/*.svg'])
datas += copy_metadata('multiphy')
datas += [(str(assets / 'licenses'), 'licenses')]
datas += [(str(assets / 'build-info.json'), '.')]

analysis = Analysis(
    [str(Path(SPECPATH) / 'cli_entry.py'), str(Path(SPECPATH) / 'gui_entry.py')],
    pathex=[str(root)], binaries=[], datas=datas,
    hiddenimports=collect_submodules('multiphy') + ['matplotlib.backends.backend_qtagg'],
    hookspath=[], hooksconfig={'matplotlib': {'backends': ['QtAgg', 'Agg']}},
    runtime_hooks=[], excludes=excluded, noarchive=False,
    module_collection_mode={'multiphy': 'pyz+py'},
)
# QtGui's hook includes every installed input-context plugin, including the
# unused GPL-only virtual keyboard. The desktop uses the native text input;
# exclude both the plugin and its library from the LGPL Qt distribution.
unused_qt = {'qtvirtualkeyboardplugin.dll', 'qt6virtualkeyboard.dll'}
analysis.binaries = [item for item in analysis.binaries
                     if Path(item[0]).name.lower() not in unused_qt]
pyz = PYZ(analysis.pure)
# Analysis includes the PyInstaller runtime hooks and both entry scripts. Each
# executable runs the common hooks followed by only its own application entry.
cli_scripts = [item for item in analysis.scripts if item[0] != 'gui_entry']
gui_scripts = [item for item in analysis.scripts if item[0] != 'cli_entry']
options = [('X utf8', None, 'OPTION'), ('u', None, 'OPTION')]
common = dict(exclude_binaries=True, debug=False, bootloader_ignore_signals=False,
              strip=False, upx=False, icon=str(assets / 'multiphy.ico'),
              version=str(assets / 'version.txt'), contents_directory='_internal')
cli = EXE(pyz, cli_scripts, options, name='multiphy', console=True, **common)
gui = EXE(pyz, gui_scripts, options, name='multiphy-gui', console=False, **common)
COLLECT(cli, gui, analysis.binaries, analysis.datas, strip=False, upx=False, name='multiphy')
