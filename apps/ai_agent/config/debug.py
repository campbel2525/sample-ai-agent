import debugpy

from config.settings import Settings

setting = Settings()

debugpy.listen(("0.0.0.0", setting.debugpy_port))
print("waiting ...")
debugpy.wait_for_client()
