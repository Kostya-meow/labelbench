# Adding a provider

1. Add `src/labelbench/providers/my_model.py` with an `AnnotationProvider` subclass.
2. Give it a stable `name`, an `availability()` check, and `annotate(image_path)`.
3. Return typed `Annotation` records. Use `bbox_xywh`; masks may be COCO RLE in
   `mask_rle`, and a polygon may be placed in `polygon`.
4. Register the class in `default_registry()`.

The registry is the only extension point used by API orchestration. Never write
outside `settings.output_dir / provider_name / run_id`.
