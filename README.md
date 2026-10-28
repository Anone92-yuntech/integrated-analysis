# 影片表情與行為整合分析

整合 `attention`、`DAN`、`medispipe_final`，以同一條 YOLOX → 官方 OC-SORT → 選定人物流程，分析同一人的玩具互動、表情、搖頭、雙手揮舞、疑似趴桌及離開座位區。三個來源專案保留；影片直接唯讀取用，不需複製。

## 啟動

先依下方「環境安裝」建立 `.venv`，執行 `fetch_models.py` 取得四個模型。已有環境與模型時，可雙擊 **`run_integrated.cmd`** 選擇影片。也可指定單支影片或資料夾（本層依檔名排序，不遞迴）：

```powershell
.\run_integrated.cmd --input 'D:\影片\受試者.mp4'
.\run_integrated.cmd --input 'D:\影片資料夾'
```

相對路徑從本專案解析，輸出預設在本專案 `logs/`。每部影片、每次執行建立唯一檔名，不覆蓋舊結果。所有計時使用來源影片時間；逐幀處理、不跳幀，處理不及時播放會變慢。暫停、輸入 ID 和手動框選不推進影片時間。不播放音訊。

```powershell
# 不顯示視窗、處理完整影片
.\run_integrated.cmd --input 'D:\影片\受試者.mp4' --headless

# 只驗證前 600 幀，同時保存逐幀診斷，方便核對
.\run_integrated.cmd --input 'D:\影片\受試者.mp4' --headless --max-frames 600 --frame-log

# 選配：匯出 1920×1080 標註影片
.\run_integrated.cmd --input 'D:\影片\受試者.mp4' --export-video
```

## 操作

使用按鍵前請將輸入法切換為英數模式；中文組字可能攔截 I／R／C／S 等按鍵。視窗內可用 Ctrl+空白切換輸入法。

| 按鍵 | 功能 |
| --- | --- |
| 空白鍵 | 暫停／繼續 |
| I | 暫停讀取並輸入可見人物 ID；Enter 確認，Backspace 修改，Esc 取消輸入 |
| R | 重新自動選人；已固定座位保留 |
| C | 立即取消座位，開放 10 秒來源影片時間供手動框選，同時開始新的穩定框校正 |
| S | 在 C 後 10 秒內手動框選；Enter 確認後立即固定，Esc 取消；框選期間影片時間停止 |
| N | 切換下一部影片，清除上一部所有人物與座位基準 |
| Q／Esc／關閉視窗 | 結束並保留已處理紀錄與片尾不足 5 秒的窗口 |

I 只能選擇當幀真正偵測配對的 OC-SORT ID，不能選灰色預測框；不需先收集肩寬。指定從下一個新影格生效，會增加 `selection_generation`、清除舊人的動作／手部歷史及坐姿基準。重選相同 ID 也視為新代次。CLI 可使用 `--target-id 2`，在第一次實際觀測到該 ID 後指定；之後漏檢仍立即重新自動選人，不會一直等原 ID。

自動選人沿用肩寬：有效樣本約 2 秒、至少 10 份，多人合格時選原影片像素肩寬中位數最小者，候選差距不足則繼續等待。這不是年齡辨識。目標缺少當幀實際偵測配對時立即重選；只有姿態不足時，約每 2 秒重建姿態模型，連續不足 10 秒才完整重選。

## 六項 5 秒輸出

固定、不重疊的來源窗口為 `[0,5)`、`[5,10)` 等，邊界影格只屬於後一窗口。每個窗口依 **`target_id`、`selection_generation` 分列**；不同人的證據不相加。窗口從中途選人、切換 ID 或重選時，該人的列可能只涵蓋部分窗口，不換算成完整 5 秒。

| 欄位 | 意義及門檻 |
| --- | --- |
| `toy_interaction` | 四類玩具 `toy/basic/toy3D/basic3D` 全部納入。任一目標手框與玩具框 IoU > 0 觸發一幀；每個來源一秒桶至少 3 幀觸發，該秒算 1 個互動秒；同一 5 秒窗口至少 3 個互動秒才為 1，否則 0。 |
| `emotion` | 0=中性、1=正向、2=無偵測。該 ID／代次列的有效表情影格須嚴格超過全部觀察影格的 50%；否則為 2，包含剛好一半。有效比例過半後，正向影格至少 5 幀為 1，否則 0。 |
| `head_shake` | 原搖頭規則成立至少 5 幀為 1，否則 0。 |
| `both_wave` | 同一幀左右手各自都符合揮舞規則，才累計雙手成立；至少 5 幀為 1，否則 0。不另輸出單手主欄。 |
| `table_rest` | 原疑似趴桌規則成立至少 5 幀為 1，否則 0。 |
| `seat_leave` | 離開座位區狀態成立至少 5 幀為 1，否則 0。 |

影格數門檻皆不要求連續；單幀有多個玩具、兩手同時碰到玩具也只計一次。玩具手框外擴 5%，沿用約 0.5 秒玩具框緩衝；雙手皆無法取得時該幀記未知，仍可用有效臉部做表情推論。玩具互動代表影像中的框重疊，不代表實體接觸或專注力。

表情只對可信目標裁切內的 Holistic 人臉進行 DAN 推論，RGB、224×224、ImageNet normalization，正向分數嚴格 > 0.5。無可信目標、無臉、臉過小或無效時為 2。沿用結果顯示時不重複累計為新推論；每個來源影格只計一次。不同 FPS 的相同影格門檻代表的秒數不同，因此保留來源 FPS 與影格計數。

搖頭／揮舞／趴桌仍先通過原本的每幀行為條件，再做 5 秒彙整：搖頭為 2 秒內至少 4 次交替極值；腕部小幅快速揮舞需 2 Hz、至少 3 次完整往返且跨度至少 1.5 秒；疑似趴桌仍先建立正常坐姿，頭下降並接近前臂連續至少 1.5 秒。五個成立影格不等於五次獨立動作。各行為可同時為 1。

## 固定座位

初次鎖定目標後，只收集當幀真正偵測配對的目標人物框；**不要求臉部或雙肩骨架**。同一身分須連續觀察至少 10 秒、至少 10 個有效框，且人物框中心相對本輪起點的歐氏距離不得超過起始框寬的 25%。剛好 25% 仍符合；超過就以目前框重新開始 10 秒。只比較中心位移，框大小變化本身不重計。

完成時對所有有效框的 `left/top/right/bottom` 分別取算術平均，固定作整部影片的座位基準。不是每幀跟隨人物移動，也不是框椅子本體。未完成校正遇漏檢、身分／代次改變或來源時間中斷會重新收集；已完成座位跨 ID、重選、R 及姿態重建保留。

按 C 後立即停用舊座位，手動窗口為 `[取消時間, 取消時間+10秒)`。按 S 所選的有效框立即固定，不需骨架或再等 10 秒；越界、黑邊、資訊欄、空框或取消不生效。若沒有手動框，仍須滿足新一輪連續穩定 10 秒才能自動建立；持續移動／漏人就繼續等待，不能在資料不足時強制平均。

座位建立後：有效人物框與座位框 IoU < 0.5 即為離開，與骨架可見性無關。已離開後，返回須 IoU ≥ 0.5、人物框中心在基準內及可信頭肩；只有框返回但骨架不足時仍等待歸席確認。未觀測到人物不算新的離座證據。改選另一人後會對同一個座位重新判斷，因此請核對輸出 ID。

## 檔案與缺資料

- `…_windows.csv`：六項主欄、窗口、ID／代次、實際觀察區間、逐項有效／未知／成立影格、互動秒數與每秒診斷。
- `…_metadata.json`：設定、模型 SHA256、版本、選人／座位操作、結束原因與總計。
- `…_frames.csv`：僅 `--frame-log` 時輸出，含每個新來源影格的證據，可獨立核對窗口。
- `…_annotated.mp4`：僅 `--export-video` 時輸出，不含音訊。

CSV 為 UTF-8 BOM，可供 Excel 開啟。玩具與身體主欄的 0 表示「未達成立門檻」，並不保證有充分資料證明未發生；請配合 `*_valid_frames`／`*_unknown_frames`。表情缺資料則直接使用 2。`unknown/calibrating/candidate` 為身體的未知證據，未確認的候選不計入成立幀。

`observed_start/end_seconds` 是實際觀察點範圍，不代表該狀態持續了整段時間。`observation_segments_json` 保留同一身分不連續的觀察片段。`Partial_Window`、`selection_partial`、`source_gap` 標示片尾、中途選人或未觀察時段；不補造缺失窗口。片尾不足 3 個互動秒時，玩具不會因縮短窗口降低門檻。

## 設定與驗證

預設讀取 `settings.json`，命令列優先。可使用 `--settings` 指定另一個平面 JSON；修改後需重新啟動。完整選項：

```powershell
.\.venv\Scripts\python.exe -B integrated_app.py --help
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B verify_windows.py 'logs\…_frames.csv' 'logs\…_windows.csv'
```

`verify_windows.py` 只讀既有 CSV，以獨立程式重新計算六項輸出，不執行模型、不改寫紀錄。驗證結果見 `VERIFICATION.md`。

## 環境安裝

本機驗證使用 Python 3.11.13、torch 2.9.1+cu128／torchvision 0.24.1+cu128。新的電腦可使用 Python 3.11 建立乾淨環境；下列安裝 CPU 版 PyTorch，不需要複製原本的 conda 或虛擬環境：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe fetch_models.py
.\.venv\Scripts\python.exe fetch_models.py --verify-only
```

模型放在本專案 `models/`；下載與 SHA256 見 `models/README.md`，程式來源紀錄見 `SOURCE_PROVENANCE.json`。公開 Git 儲存庫僅包含程式、設定、測試與文件；來源影片、人物影格、逐幀預測、CSV、執行紀錄及本機環境不隨專案提供。請自行準備輸入影片。

官方 OC-SORT 與 DAN 授權隨程式保留，第三方來源見 `THIRD_PARTY_NOTICES.md`。此整合版只推論、不重訓、不讀取或複製共享資料集。後續若增加訓練，資料仍須直接使用 `Path(__file__).resolve().parent.parent / "emotion_dataset"` 的原切分及 `0=neutral、1=positive` 標籤；2 僅為影片推論缺資料代碼，不新增訓練類別。

現有測試驗證程式規則與可執行性；尚未以人工標註量測整合後的影片辨識準確率。追蹤 ID 可能交換，肩寬選人可能選錯，表情不等同內在情緒，身體動作也不直接代表不耐煩。
