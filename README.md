# YOLOv5 Defect Detection Web App 缺陷偵測網頁系統

本專案是一個以 Flask 架設的 YOLOv5 推論平台，可上傳圖片與影片進行物件偵測，支援同步播放並即時顯示辨識結果。

This project is a YOLOv5 inference web application built with Flask. Users can upload images or videos to perform object detection, with results displayed directly on the webpage.

---

## 專案結構 Project Structure

```
project_root/
├── static/                # 靜態資源（上傳檔案與輸出結果）
│   └── uploads/           # 儲存使用者上傳的圖片或影片
├── templates/             # HTML 模板（主要為 index.html）
├── yolov5/                # YOLOv5 模型資料夾
├── val/                   # 驗證資料集（非必須）
├── best.pt                # 訓練好的 YOLOv5 權重檔
├── app.py                 # 主程式 Flask 應用
└── README.md              # 使用說明文件
```

---

## 快速啟動 Quick Start

### 1 安裝套件 Install dependencies

```bash
pip install flask torch torchvision opencv-python pillow
```

> 若使用的是 YOLOv5 原始碼，建議依照其 `requirements.txt` 安裝。

---

### 2 啟動伺服器 Run the Flask App

```bash
python app.py
```

預設會啟動在 `http://127.0.0.1:5002`。

Default URL: `http://127.0.0.1:5002`

---

## 圖片偵測 Image Detection

- 上傳一張圖片（支援 JPG、PNG 等格式）
- 後端將使用 YOLOv5 模型進行推論
- 顯示偵測結果圖片與物件數量統計

---

## 影片偵測 Video Detection

- 上傳影片檔案（支援 `.mp4`, `.avi`, `.mov`）
- 系統逐幀進行推論，並即時渲染框圖
- 回傳推論後的影片檔與偵測類別數量

---

## API 端點 API Endpoints

| 路徑 Path            | 方法 Method | 說明 Description                  |
|---------------------|--------------|----------------------------------|
| `/`                 | GET          | 首頁，顯示上傳表單與偵測介面      |
| `/detect`           | POST         | 圖片偵測（上傳圖片）              |
| `/video_detect`     | POST         | 影片偵測（上傳影片）              |
| `/detect_frame`     | POST         | 即時單張畫面推論（回傳圖片）     |
| `/get_image`        | GET          | 取得偵測結果圖片（for 顯示）      |

---

## 模型設定 Model Setup

- 使用自訂訓練好的 YOLOv5 權重檔 `best.pt`
- 模型信心閾值設定為 `0.25`（可於 `app.py` 調整）

```python
model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')
model.conf = 0.25
```

---

## 注意事項 Notes

- 上傳檔案將儲存於 `static/uploads/` 中，請定期清理。
- 僅限支援單模型 `best.pt`，如需切換類別或架構請重新訓練。
- 影片推論需較長時間，請耐心等待處理完成。

---

## 未來可擴充功能 To-Do

- 分類數量統計顯示於介面
- 即時影片串流辨識
- 部署至 Heroku、GCP、Fly.io
- 包裝成桌面應用（EXE）
