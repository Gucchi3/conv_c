cmake_minimum_required(VERSION 3.10)
# プロジェクト名
project(MyApp)

# すべてのソースファイルを収集
file(GLOB_RECURSE SOURCES "C_core/src/**/*.c")

# utf8 対応
add_compile_options(/utf-8)

# 実行ファイル名、コンパイル対象ファイル選択
add_executable(my_app ./C_core/main.c ${SOURCES} ./generated/weight.c)

###################################　
target_compile_definitions(my_app PRIVATE DEBUG=1)
###################################


# インクルードディレクトリを指定
target_include_directories(my_app PRIVATE C_core/include)