"""Demo Gallery：可插拔 Demo 的注册中心与挂载层。

加一个 Demo = 在 ``demos/<id>/`` 建一个文件夹，用 ``@demo`` 装饰器注册。
**不需要修改本包或 gateway.py 的任何一行。**

规范见 ``demos/README.md``。
"""

from demo_gallery.mounting import install_gallery
from demo_gallery.registry import DemoSpec, StaticMount, demo, registry

__all__ = ["DemoSpec", "StaticMount", "demo", "install_gallery", "registry"]
