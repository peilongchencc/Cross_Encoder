# Cross_Encoder

本项目用于介绍`sentence_transformers`库中`CrossEncoder`在文本相似度方面的使用。
- [Cross\_Encoder](#cross_encoder)
  - [文件简介:](#文件简介)
  - [CrossEncoder简介:](#crossencoder简介)
    - [CrossEncoder与常规查询相似文本方式的不同:](#crossencoder与常规查询相似文本方式的不同)
    - [总结](#总结)
  - [依赖项安装:](#依赖项安装)
  - [启动主程序:](#启动主程序)
  - ["CrossEncoder.rank" 参数详解:](#crossencoderrank-参数详解)
  - [Debug:](#debug)
  - [注意事项:](#注意事项)

## 文件简介:

| 文件夹名称                        | 作用                                    | 备注                        |
|---------------------------------|-----------------------------------------|-----------------------------|
| main.py                         | 常规版本主程序                             | 通过FastAPI启动服务          |
| main_multi.py                   | 多进程版本主程序                           | 通过FastAPI启动服务           |
| test_script.py                  | 测试脚本                                  |                             |
| rank_documents.py               | 相似度计算                                | 工具函数                      |
| regular_cross_encoder.py        | 常规示例                                  |                             |
| multiprocessing_cross_encoder.py| 多进程版本示例                             |                              |


## CrossEncoder简介:

`CrossEncoder`是用于 **"文本对"** 任务的模型，比如分类、回归、以及计算两个句子的相似度。它通过将两个句子拼接在一起，然后将拼接后的输入传递给一个预训练的Transformer模型（如BERT、RoBERTa）来直接进行预测。

### CrossEncoder与常规查询相似文本方式的不同:

在查询相似文本任务中，常用的方法是`Bi-Encoder`。`Bi-Encoder`和`CrossEncoder`的主要区别在于它们处理句子对的方式不同：

1. **Bi-Encoder**:

   - 在`Bi-Encoder`模型中，句子1和句子2分别独立地传递给两个相同的Transformer模型，产生它们各自的向量表示（即嵌入）。

   - 这些嵌入可以被存储、索引，之后只需计算它们之间的相似度（例如余弦相似度）即可判断文本的相似性。

   - 由于计算句子嵌入是独立的，`Bi-Encoder`非常适合大规模语料库的相似性搜索（如语义检索），因为它允许提前对语料库中的所有句子进行编码并存储，查询时只需一次前向传播。

2. **CrossEncoder**:

   - 在`CrossEncoder`模型中，句子1和句子2被同时输入模型(拼接句子形成一个整体的编码，然后转为词向量)，它能够充分利用两者之间的相互关系来计算得分。

   - `CrossEncoder`不生成独立的句子嵌入，因此不适合用在需要快速计算大规模语料库中多个句子对相似性的任务。

   - 由于每次比较都需要重新计算两句话的表示，因此这种方法通常比`Bi-Encoder`计算代价更高，但在准确性上往往更强。

### 总结

- `CrossEncoder` 通过直接输入句子对，并利用Transformer模型来同时处理这两个句子的关系，从而生成一个得分。它在精确性上通常优于`Bi-Encoder`，但计算效率较低。

- `Bi-Encoder` 适合在大规模检索场景下使用，因为它能够独立生成每个句子的嵌入，然后快速进行相似性比较。


## 依赖项安装:

1. 安装python环境:

笔者使用的python 3.12，安装方式如下:

```bash
conda create -n cross_encoder python
```

2. 激活虚拟环境:

```bash
conda activate cross_encoder
```

3. 安装常规依赖项:

```bash
pip install -U sentence-transformers
pip install "fastapi[standard]"
pip install aiohttp
pip install loguru
```


## 启动主程序:

终端输入以下指令(默认端口号为8000)启动服务:

```bash
fastapi dev main.py
```

如果你想要使用其他端口，可参考以下指令:

```bash
fastapi dev main.py --port 8800
```

🚨以上方式不允许外网访问，如果想要从外网访问，可参考以下形式:

```bash
fastapi dev main.py --host 0.0.0.0 --port 8800
```


## "CrossEncoder.rank" 参数详解:

```conf
执行 CrossEncoder 对给定查询和文档的排名。返回一个按文档索引和分数排序的列表。

参数:
    query (str): 单个查询。
    documents (List[str]): 文档列表。
    top_k (Optional[int], 可选): 返回前 k 个文档。如果为 None，则返回所有文档。默认为 None。
    return_documents (bool, 可选): 如果为 True，还会返回文档。如果为 False，则只返回索引和分数。默认为 False。
    batch_size (int, 可选): 编码的批大小。默认为 32。
    show_progress_bar (bool, 可选): 显示进度条输出。默认为 None。
    num_workers (int, 可选): 用于分词的工作线程数量。默认为 0。
    activation_fct ([type], 可选): 应用于 CrossEncoder 输出的 logits 的激活函数。如果为 None 且 num_labels=1，则使用 nn.Sigmoid()，否则使用 nn.Identity。默认为 None。
    convert_to_numpy (bool, 可选): 将输出转换为 numpy 矩阵。默认为 True。
    apply_softmax (bool, 可选): 如果维度大于 2 并且 apply_softmax=True，则对 logits 输出应用 softmax。默认为 False。
    convert_to_tensor (bool, 可选): 将输出转换为 tensor。默认为 False。

返回:
    List[Dict[Literal["corpus_id", "score", "text"], Union[int, float, str]]]: 一个按文档的 "corpus_id"、"score" 和可选的 "text" 排序的列表。
```


## Debug:

FastAPI程序进行Debug需要将代码格式写为`if main`形式，例如:

```python
import uvicorn
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    a = "a"
    b = "b" + a
    return {"hello world": b}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Debug时，如果想要深入代码内部，可以参考以下`launch.json`内容:

```json
{
    // 使用 IntelliSense 了解相关属性。 
    // 悬停以查看现有属性的描述。
    // 欲了解更多信息，请访问: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python 调试程序: 当前文件",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": false
        }
    ]
}
```


## 注意事项:

🚨在使用 Python 的多进程模块时，一定遵循正确的模块导入模式，错误的导入方式会导致程序无法启动。