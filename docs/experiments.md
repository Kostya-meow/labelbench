# Эксперименты в LabelBench

## Быстрый запуск

1. Запусти `bat/START.bat`, открой `http://127.0.0.1:8000`.
2. Слева выбери `архив 2`. Если список пуст — «Подключить папку с GT», указать папку с `dataset.json`.
3. Выбери 2–3 текстовые модели и confidence. Пустое поле означает штатный порог модели.
4. «Страница»: запустить модели, включать/выключать классы и цвета, наложить эталон пунктиром, сравнить с GT, отправить кандидатов в VLM.
5. «Эксперименты»: название, dev/test, seed, лимит страниц, IoU и метод объединения. Начни с 10 dev-страниц. Состояние, время, ошибки и завершённые страницы сохраняются.
6. Из истории открой метод и страницу, посмотри полигонные результаты. JSON-отчёт содержит настройки и метрики. Для собственного метода загрузить JSON из notebooks с `image_name` и `annotations`.

## Пороги

PP-OCRv5/v6: порог уверенности области (`box_thresh` / `box_threshold`), не порог бинаризации карты. YOLO, RF-DETR, Mask2Former: confidence детекции. SAM: predicted IoU; stability threshold остаётся 0.92. Doc-UFCN, Eynollah, RTMDet: фильтр объектов после штатного decoding. Снижение такого фильтра **не восстанавливает** отброшенные внутри модели предложения. Confidence разных семейств не калиброван на общей шкале.

Настройки входят в ключ кэша. Для изменённого confidence вычисляется отдельный результат; старый не затирается. В экспериментах неподписанные legacy-запуски не используются. Отдельный запуск страницы сохраняет PNG, пакетный — компактные JSON и кэш, без тысяч дублирующих PNG.

## API

Интерактивная схема: `/docs`, машиночитаемая: `/openapi.json`. Существующие `/api/runs`, `/api/llm/review` совместимы с прежними запросами.

```python
import requests

api = "http://127.0.0.1:8000/api"
dataset = requests.post(f"{api}/datasets", json={
    "root": r"D:\Разные разметки\архив 2"
}).json()

job = requests.post(f"{api}/experiments", json={
    "name": "tiny + RF-DETR, dev, confidence 0.5",
    "dataset_id": dataset["id"],
    "providers": ["ppocr6_tiny", "rfdetr_historical"],
    "options": {"ppocr6_tiny": {"confidence": 0.5}},
    "split": "dev", "seed": 42, "limit": 10,
    "methods": ["union", "nms", "consensus"],
    "match_iou": 0.5, "fusion_iou": 0.5, "min_votes": 2
}).json()
status = requests.get(f"{api}/experiments/{job['id']}").json()
```

| Endpoint | Назначение |
|---|---|
| GET/POST `/api/datasets` | Список / регистрация архива без копирования изображений |
| GET `/api/datasets/{id}` | Файлы, отсутствующие изображения, проверка полигонов, версия GT |
| GET `/api/datasets/{id}/truth?name=109.jpg` | Эталонные полигоны страницы |
| POST `/api/runs` | `image_name`, `providers`, необязательные `dataset_id`, `options`, `force`, `progress_id` |
| POST `/api/runs/{id}/evaluate` | `iou`, `text_only`; метрики по каждому провайдеру |
| GET/POST `/api/experiments` | История / фоновый эксперимент; запуск возвращает HTTP 202 |
| GET `/api/experiments/{id}` | Статус, elapsed, события, summary и ошибки |
| POST `/api/experiments/{id}/cancel` | Остановка между inference: текущий GPU-вызов не прерывается |
| GET `/api/experiments/{id}/pages/{index}` | Предсказания и метрики, индекс с нуля |
| POST `/api/experiments/{id}/pages/{index}/open?method=fusion:nms` | Сохранить выбранный результат как run для UI/VLM |
| POST `/api/evaluations` | Оценить свой метод: `dataset_id`, `image_name`, `method`, `annotations`, `iou`; polygon в пикселях, bbox не нужен |

`annotations`: `[{"polygon":[[0,0],[10,0],[10,10],[0,10]],"score":0.9,"label":"text"}]`.

## Что именно сохраняется

`data/output/datasets`: manifest и SHA-256 эталона; `experiments/<id>/experiment.json`: конфигурация, выбранные страницы, версии пакетов, git revision и хэш Python-исходников; `pages/<index>.json`: предсказания всех методов и TP/FP/FN, matching. В run есть хэш изображения и ключи кэша моделей. `reviews`: параметры и ответы VLM, метрики при доступном GT; API-ключ исключён. Пользовательские данные и эти результаты исключены из Git.

Перезапуск сервера помечает незавершённый эксперимент как interrupted. Для продолжения создаётся новый запуск с теми же настройками; завершённые детекции возьмутся из кэша. Только один пакетный эксперимент одновременно, inference сериализован для защиты GPU. Запускать один процесс сервера, без нескольких uvicorn workers.

## Метрики и ограничения

Оценка class-agnostic для выбранных кандидатов, exact planar polygon IoU, однозначное matching. Дубликаты — FP; отсутствующие/невалидные полигоны предсказаний — unmatched FP. Некорректный GT вызывает явную ошибку страницы. Пустая страница имеет F1=0 и флаг empty_page; mean_matched_iou=null без совпадений. Ошибочные страницы перечисляются отдельно и не входят в summary: сравнивать методы только при одинаковом покрытии.

SAM/Mask2Former сохраняют полный RLE и крупнейший внешний контур для полигонного API. Для самокасающихся масок YOLO/RF-DETR контур исправляется и выбирается крупнейшая валидная компонента. Полигонная оценка не эквивалентна оценке полного RLE с отверстиями/раздельными компонентами. Для текстовых фрагментов это нужно учитывать в постановке задачи.

NMS использует исходный confidence. Consensus считает разные provider IDs и выбирает существующий полигон с максимальным согласием; никаких усреднённых прямоугольников. Несколько вариантов PP-OCR не означают независимые ошибки моделей. Показатель «Согласие» на странице — эвристика среднего лучшего IoU с другой моделью, **не точность**.

## ONNX

Eynollah уже работает через отдельный ONNX GPU worker. Добавлен YOLO26 ONNX как отдельный provider `yolo26_onnx`; исходный `yolo26` сохранён. Подготовка:

```bat
bat\EXPORT_YOLO_ONNX.bat "путь-к-изображению-с-объектами.jpg"
```

Экспорт YOLO: FP32, 640×640, opset 17. Проверка на CUDA с тем же preprocessing и polygon IoU >=0.95. На пустых предсказаниях проверка не считается успешной. Наличие CUDAExecutionProvider проверяется после inference; молчаливое переключение всего backend на CPU считается ошибкой. Это smoke parity на одном изображении, не доказательство одинакового качества и скорости на всём архиве.

PP-OCRv6 Medium/Small/Tiny: `bat\EXPORT_PPOCR6_ONNX.bat "изображение.jpg"`.
API IDs: `ppocr6_onnx`, `ppocr6_small_onnx`, `ppocr6_tiny_onnx`.
Экспорт сохраняет динамические размеры изображения, официальный preprocessing и декодирование полигонов.
Doc-UFCN: `bat\EXPORT_DOCUFCN_ONNX.bat "изображение.jpg"`, API ID `docufcn_onnx`.
Для Doc-UFCN BatchNorm без накопленных статистик записывается эквивалентными операциями mean/variance: ONNX Runtime CUDA не поддерживает этот training-mode BatchNorm напрямую. Итоговые полигоны проверяются против исходной модели.
В интерфейсе backend выбирается под соответствующей моделью. Непроверенный экспорт недоступен.
RF-DETR: `bat\EXPORT_RFDETR_ONNX.bat "изображение.jpg"`, API ID `rfdetr_historical_onnx`. Сравнение с FP32 inference-режимом исходной модели, preprocessing и segmentation postprocessing остаются из RF-DETR. Основная сеть работает через ONNX CUDA, postprocessing — на CPU. Не устанавливайте CPU-пакет `onnxruntime` поверх `onnxruntime-gpu`.
Для RTMDet, PP-OCRv5, SAM2 и Mask2Former пока используется исходный backend; ONNX для них здесь не заявлен как готовый.

## Повторная конвертация архива из базы

`dataset.json` и `annotations.db` в исходном архиве — разные версии. На просмотренных страницах JSON содержит слова, а база содержит полигоны строк внутри колонок. Для оценки строк выбирайте **Archive DB — line polygons**, а не старый импорт JSON.

```powershell
.venv\Scripts\python.exe scripts\convert_archive.py "архив 2" "data/output/archive-lines-db-v2"
```

Команда делает согласованный SQLite snapshot, включая WAL, в новой папке; оригинал не изменяется, фотографии не дублируются. Каждый полигон базы сохраняется отдельно, координаты 0–1 переводятся в пиксели. `line_id` не используется для склейки: одна группа может включать несколько физических строк.

Проверенный снимок `archive-lines-db-v1`: 977 доступных страниц, 48 790 полигонов; 23 страницы исключены полностью из-за 153 объектов без валидной сегментации. 26 имён из базы отсутствуют в архиве. Отчёт с точными именами: `data/output/archive-lines-db-v1/conversion-report.json`. Поворот просмотра на 2° у `112.jpg` записан в отчёте; координаты остаются относительно исходного изображения. Геометрическая проверка выполнена для всех объектов, визуальная проверка — выборочная, не экспертная аттестация всего GT.

План научных сравнений: [research-2026.md](research-2026.md).
