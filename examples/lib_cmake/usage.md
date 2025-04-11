# 打包为动/静态库

本节提供模板库算子编译成动/静态库，并在已有工程中调用的cmake配置示例.

用户可使用或参考cmake_modules/aux_functions.cmake配置编译动/静态库的cmake工程.

## 代码结构

```
├── examples
│   └── lib_cmake
│       ├── CMakeLists.txt
│       ├── impl
│       │   ├── CMakeLists.txt
│       │   ├── cmake_modules
│       │   │   └── aux_functions.cmake // 样例使用cmake构建动/静态库时用到的辅助函数
│       │   └── matmul_add.cpp
│       ├── main.cpp
│       └── usage.md
└── scripts
    └── build.sh
```

### 使用CMake打包动/静态库示例

```bash
bash scripts/build.sh lib_cmake
```

## 编译产物结构

```bash
├── bin
│   └── QuickStart
└── examples
    └── lib_cmake
        └── impl
            ├── libascend_device.a  # bisheng_add_library(ascend_device STATIC ${LIB_SOURCES})
            └── libascend_device.so # bisheng_add_library(ascend_device DYNAMIC ${LIB_SOURCES})
```

## 注意事项
- 本节是算子打包成动/静态库的一个示例，可根据需要自行扩展功能.
