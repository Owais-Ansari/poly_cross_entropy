# poly_cross_entropy

Poly cross entropy for segmemtation


***reference***
https://arxiv.org/pdf/2204.12511.pdf

***Usage***
```shell script
from loss import poly_cross_entropy1

loss = poly_cross_entropy1(outputs,targets.long(),num_classes=num_classes, ignore_label = num_classes, gpu=gpu)

```
