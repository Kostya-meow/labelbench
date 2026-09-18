# LabelBench — локальная авторазметка изображений

Локальный API и браузерный интерфейс для сравнения предсказаний семи независимых
моделей: **PP-OCRv5 Server**, **Mask2Former**, **SAM 2.1**, **YOLO26-seg**, **RF-DETR Historical Textline**,
**Doc-UFCN Generic Historical Line** и **Eynollah Textline**.
Новые модели добавляются
одним адаптером, не меняя API и интерфейс.

## Что умеет

- читает изображения из `data/images/`;
- сохраняет каждый запуск отдельно: `data/output/<provider>/<run-id>/`;
- хранит унифицированный COCO-подобный JSON с боксами, полигонами/RLE-масками,
  метками, confidence и происхождением;
- показывает исходник, наложения моделей и предварительный consensus score;
- после inference позволяет для каждого provider-а включать/выключать результат, выбирать цвет каждого класса сегмента и скачать оставленные аннотации отдельным JSON;
- отправляет выбранные результаты вместе с изображением в локальный VLM через LM Studio и показывает улучшенную разметку отдельным слоем;
- работает без загруженных ML-пакетов: интерфейс запускается, а статус сразу
  объясняет, какой provider недоступен и почему;
- загружает веса при первом обращении в стандартный кэш Hugging Face/PaddleOCR.

## Быстрый старт (Windows)

Нужны Python 3.10, Git и `uv` (`winget install Astral-sh.uv`). Последовательно запустите:

1. `bat\INSTALL_GPU.bat`
2. `bat\DOWNLOAD_WEIGHTS.bat`
3. `bat\START.bat`

Установщик создаёт отдельные окружения `.venv-gpu` и `.venv-ocr`, чтобы CUDA-библиотеки
PyTorch и Paddle не конфликтовали в одном Windows-процессе.

Откройте <http://127.0.0.1:8000>, скопируйте изображения в `data/images/` и нажмите
«Запустить выбранные». Веса заранее скачивает `bat\DOWNLOAD_WEIGHTS.bat`.

### Проверка разметки через LM Studio

В LM Studio запустите локальный сервер на `http://localhost:1234` и загрузите
основную модель **PaddleOCR VL 1.6 GGUF**. Файл **Mmproj GGUF** подключите в
диалоге загрузки как projector. Сам файл `mmproj` нельзя выбирать как основную
модель: он является CLIP/projector-файлом. Ошибка `CLIP cannot be used as main
model, use it with --mmproj instead` означает, что выбран именно `mmproj`.
В LabelBench основная модель должна появиться с идентификатором
`paddleocr-vl-1.6`. После обычного inference выберите нужные
provider-ы, задайте промпт и нажмите «Отправить в VLM». В запросе передаются
исходное изображение, результаты выбранных моделей и история текущего диалога.

Координаты имеют единый формат: начало в левом верхнем углу изображения,
`bbox_xywh: [x, y, width, height]` в пикселях, polygon — список пар `[x, y]`.
VLM возвращает JSON-действия `keep`, `remove`, `modify`, `add`; приложение
проверяет границы, применяет их к исходным объектам и позволяет скачать
`*_vlm.json`.

Если модель видна в списке, но при отправке появляется `Failed to load model`,
загрузите её кнопкой загрузки модели в LM Studio и дождитесь окончания загрузки.
Список в `/v1/models` подтверждает наличие модели, но не гарантирует, что она
уже загружена в VRAM.

> **SAM 2.1:** для GPU-инференса Meta рекомендует WSL2/Linux; скрипт всё равно
> поддерживает Windows, но CUDA extension может не собраться. Базовый inference
> обычно продолжает работать без extension, с ограниченным post-processing.

## Самостоятельная установка через BAT

Для ручной установки без фоновых задач в папке `bat/` есть три файла:

1. `INSTALL_GPU.bat` — создаёт окружение и проверяет работающую CUDA у PyTorch и Paddle.
2. `DOWNLOAD_WEIGHTS.bat` — скачивает веса всех моделей только после успешной GPU-проверки.
3. `START.bat` — поднимает интерфейс на `http://127.0.0.1:8000`.

Нужны установленный `uv`, Git (для SAM 2) и доступ к официальным индексам PyTorch,
PaddlePaddle, GitHub, Hugging Face и Paddle model hosting. Каждый BAT можно
повторно запускать после обрыва сети.

## Providers и модели по умолчанию

| Provider | Роль | Вес по умолчанию |
| --- | --- | --- |
| `ppocr` | текстовые области и распознанный текст | PP-OCRv5 Server (PaddleOCR 3.x) |
| `mask2former` | panoptic/instance-кандидаты классов COCO | `facebook/mask2former-swin-large-coco-panoptic` |
| `sam2` | class-agnostic object masks | `facebook/sam2.1-hiera-large` |
| `yolo26` | COCO instance segmentation | `yolo26n-seg.pt` |
| `rfdetr_historical` | historical text lines and text regions | `Kansallisarkisto/rfdetr_textline_textregion_detection_model` |
| `docufcn` | generic historical text lines | `Teklia/doc-ufcn-generic-historical-line` |
| `eynollah_textline` | historical textline segmentation | [`SBB/eynollah-textline`](https://huggingface.co/SBB/eynollah-textline) |

Для исторического **Mask R-CNN** отдельный публичный checkpoint с воспроизводимым inference API
не найден, поэтому COCO/обычный scene-text checkpoint сюда не подставлен.

Настройки задаются переменными окружения: `LABELBENCH_DEVICE` (`auto`, `cpu`,
`cuda`), `LABELBENCH_MASK2FORMER_MODEL`, `LABELBENCH_SAM2_MODEL`,
`LABELBENCH_SAM2_POINTS_PER_SIDE`, `LABELBENCH_YOLO26_MODEL`,
`LABELBENCH_RFDETR_CHECKPOINT`, `LABELBENCH_DOCUFCN_CHECKPOINT`,
`LABELBENCH_EYNOLLAH_CHECKPOINT`, `LABELBENCH_EYNOLLAH_PYTHON`.
Для LM Studio доступны `LABELBENCH_LM_STUDIO_URL` (по умолчанию
`http://localhost:1234/v1`), `LABELBENCH_LM_STUDIO_API_KEY` и
`LABELBENCH_LM_STUDIO_TIMEOUT`.

## API

- `GET /api/health` — доступность providers.
- `GET /api/images` — список входных изображений.
- `POST /api/runs` — запустить одно изображение: `{"image_name":"a.jpg","providers":["ppocr","sam2"]}`.
- `GET /api/runs/{run_id}` — объединённый результат.
- `GET /api/runs/{run_id}/overlay/{provider}` — наложение provider-а.
- `GET /api/llm/models` — список моделей, опубликованных LM Studio.
- `POST /api/llm/review` — отправить изображение и выбранные результаты в VLM;
  ответ содержит `refined_annotations`, `notes` и исходный ответ модели.

## Как добавить ещё одну модель

Создайте класс в `src/labelbench/providers/`, наследующий `AnnotationProvider`,
и зарегистрируйте его в `registry.py`. Он возвращает `ProviderResult` и поэтому
сразу становится доступен в API, в папке output и в UI. Детали — в
[`docs/adding_provider.md`](docs/adding_provider.md).

## Автоматическая разметка: как интерпретировать результат

PP-OCR и Mask2Former делают независимые предсказания. SAM 2.1 формирует
автоматические proposal-маски сеткой point prompts; это не семантические классы.
`consensus_score` — пространственное согласие с другими кандидатами, а не
истинная точность. Результаты предназначены для ускорения проверки человеком,
а не для автоматического объявления ground truth.
