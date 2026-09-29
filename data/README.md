# AgRobTomato data

Download the archive from <https://zenodo.org/records/5596799> and keep it under `data/agrob/`. Its MD5 is `890666716924415720f073b06a9a02a3`. Extract it so `Annotations/` and `JPEGImages/` are directly inside `data/agrob/Dataset-Greenhouse_Tomato_AgRob/`.

The source archive contains 449 JPEG images and 449 Pascal VOC XML annotations. The four ripeness labels describe individual fruit. It has no truss labels. `evaluate_agrob_fruit.py` reads this layout directly; there is no conversion step for the current baseline.

Downloaded images and generated model output are ignored by Git. The tracked `results/agrob/` folder contains only small summaries and per-image count tables. Cite the Zenodo record when sharing results.
