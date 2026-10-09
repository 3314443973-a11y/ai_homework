# analyze text 接口约定

## 输入

`analyze(text)` 接收一段非空校园诉求文本。

## 返回值

```json
{
  "label": "宿舍设施",
  "urgency": "高",
  "department": "慧湖物业中心",
  "summary": "学生反映被子被锁在室外，夜间缺少基本御寒用品。"
}
```

## 字段要求

- `label`：六个固定类别之一。不得返回 `category`。
- `urgency`：高、中、低。
- `department`：由最终 `label` 映射产生。
- `summary`：一至两句规范化摘要，失败时使用清理后的原文。

旧代码若仍读取 `category`，应在联调时一次性改为读取 `label`。正式接口不同时保留两个同义字段，避免前后端长期混用。
