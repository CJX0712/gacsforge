"""错误体系 E100~E500。作者: 晨星"""


class SemiForgeError(Exception):
    code = "E000"

    def __init__(self, message: str) -> None:
        super().__init__(f"[{self.code}] {message}")


class ConfigError(SemiForgeError):
    code = "E100"


class DataError(SemiForgeError):
    code = "E200"


class MethodError(SemiForgeError):
    code = "E300"


class EvalError(SemiForgeError):
    code = "E400"


class PipelineError(SemiForgeError):
    code = "E500"
