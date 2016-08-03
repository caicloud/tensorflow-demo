py_library(
    name = "hello_lib",
    srcs = [
        "hello_lib.py",
    ]
)

py_binary(
    name = "hello_main",
    srcs = [
        "hello_main.py",
    ],
    deps = [
        ":hello_lib",
    ],
)

