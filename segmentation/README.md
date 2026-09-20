# segmentation

把扫描版 PDF 表格按类型拆分成单条记录、并转正翻转页面的脚本。

## PDF 放哪里

在这个 `segmentation` 文件夹**外面**(同级)新建一个叫 `pdf` 的文件夹,把要处理的 PDF 文件直接放进去:

```
JRS-IQ/              <- 仓库根目录
├── pdf/              <- 新建这个文件夹,PDF 直接放这里(不进 git,不用提交)
│   ├── xxx.pdf
│   └── yyy.pdf
└── segmentation/      <- 这个仓库里已有的代码
    └── program/
```

程序默认就是从 `segmentation/program/` 往上两级找 `pdf/` 文件夹,不用改代码、不用传参数。

## 环境准备

```
cd segmentation
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## 怎么跑

```
cd segmentation\program
python stage1_discover.py          # 第一步:扫描全部 PDF,产出待人工标注的 discovery_full/
```

跑完打开 `discovery_full/clusters/` 里的每张图看一眼,把 `discovery_full/labels_template.csv` 填好(每个图对应一行,填 角色/表号/需要转的角度),另存成同目录下的 `labels.csv`,再跑:

```
python stage2_segment.py           # 第二步:按标注结果拆分、转正、出结果
```

结果在 `../../segmented_form/<表号>/` 下,附一份 `manifest.csv` 核对清单。

每个脚本文件开头的注释里有更详细的字段说明。
