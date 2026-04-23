# boundary_gpt vs softmax vs ord2seq 对比报告

## VAL 集

### boundary_gpt 相邻错级 (n=174)

| image_id | roi_id | ic4 | sm_pred | ord_pred | bd_pred |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0b700680-102.jpg | 4 | A | 0 | 0 | 0 |
| 0b700680-102.jpg | 5 | A | 0 | 0 | 0 |
| 2bfaa34c-259.jpg | 4 | 0 | A | 0 | A |
| 2bfaa34c-259.jpg | 7 | 0 | A | A | A |
| 2bfaa34c-259.jpg | 9 | A | 0 | 0 | 0 |
| 5ff09f48-anonymous_003_007_366_01_1728116583995_Maxillary_Occlusal_View.jpg | 1 | C | B | B | B |
| 5ff09f48-anonymous_003_007_366_01_1728116583995_Maxillary_Occlusal_View.jpg | 2 | C | B | B | B |
| f0daeb4d-28.jpg | 1 | B | B | A | A |
| f0daeb4d-28.jpg | 9 | B | B | B | A |
| f0daeb4d-28.jpg | 10 | B | A | A | A |
| f0daeb4d-28.jpg | 11 | A | A | A | 0 |
| f0daeb4d-28.jpg | 12 | B | A | A | A |
| f0daeb4d-28.jpg | 15 | 0 | 0 | A | A |
| f0daeb4d-28.jpg | 18 | 0 | A | A | A |
| f887a525-273.jpg | 11 | A | 0 | A | 0 |
| f887a525-273.jpg | 14 | A | 0 | 0 | 0 |
| f887a525-273.jpg | 15 | A | A | 0 | 0 |
| f887a525-273.jpg | 16 | A | A | A | 0 |
| f9a8221b-194.jpg | 3 | B | B | A | A |
| f9a8221b-194.jpg | 4 | B | B | B | A |
| f9a8221b-194.jpg | 8 | 0 | A | A | A |
| f9a8221b-194.jpg | 10 | 0 | A | 0 | A |
| 03e5a7c5-115.jpg | 11 | 0 | A | A | A |
| 03e5a7c5-115.jpg | 12 | 0 | A | A | A |
| 03e5a7c5-115.jpg | 14 | 0 | 0 | A | A |
| 03e5a7c5-115.jpg | 21 | A | A | A | 0 |
| 03e5a7c5-115.jpg | 23 | 0 | A | A | A |
| 03e5a7c5-115.jpg | 25 | 0 | A | A | A |
| 03e5a7c5-115.jpg | 27 | 0 | A | A | A |
| 054ab6ae-3.jpg | 1 | A | B | B | B |
| 054ab6ae-3.jpg | 11 | B | B | A | A |
| 1131e1c7-78.jpg | 0 | A | B | A | B |
| 1131e1c7-78.jpg | 3 | 0 | A | A | A |
| 1131e1c7-78.jpg | 4 | 0 | A | 0 | A |
| 1131e1c7-78.jpg | 8 | A | A | 0 | 0 |
| 1131e1c7-78.jpg | 10 | 0 | A | A | A |
| 1355aad5-244.jpg | 0 | 0 | A | A | A |
| 1355aad5-244.jpg | 5 | A | A | 0 | 0 |
| 1355aad5-244.jpg | 8 | A | 0 | A | 0 |
| 27e4fdf6-235.jpg | 1 | A | B | B | B |

### boundary_gpt 跨两级及以上错误 (n=6)

| image_id | roi_id | ic4 | sm_pred | ord_pred | bd_pred |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 03e5a7c5-115.jpg | 15 | B | 0 | 0 | 0 |
| 52e40dc5-120.jpg | 15 | 0 | 0 | B | B |
| cbdcf4fe-16.jpg | 19 | B | 0 | 0 | 0 |
| cbdcf4fe-16.jpg | 20 | B | A | A | 0 |
| cbdcf4fe-16.jpg | 21 | B | A | 0 | 0 |
| cbdcf4fe-16.jpg | 22 | B | 0 | 0 | 0 |

### boundary_gpt 错, softmax+ord2seq 都对 (n=21)

| image_id | roi_id | ic4 | sm_pred | ord_pred | bd_pred |
| :--- | :--- | :--- | :--- | :--- | :--- |
| f0daeb4d-28.jpg | 9 | B | B | B | A |
| f0daeb4d-28.jpg | 11 | A | A | A | 0 |
| f887a525-273.jpg | 16 | A | A | A | 0 |
| f9a8221b-194.jpg | 4 | B | B | B | A |
| 03e5a7c5-115.jpg | 21 | A | A | A | 0 |
| 27e4fdf6-235.jpg | 7 | 0 | 0 | 0 | A |
| 2824f032-264.jpg | 11 | A | A | A | 0 |
| 2eca70d7-99.jpg | 11 | B | B | B | A |
| 48aa0fe4-183.jpg | 8 | A | A | A | 0 |
| 88e305ce-74.jpg | 7 | A | A | A | 0 |
| 9667fc01-139.jpg | 6 | 0 | 0 | 0 | A |
| 9667fc01-139.jpg | 20 | 0 | 0 | 0 | A |
| 9d58e82c-249.jpg | 16 | A | A | A | 0 |
| 9d58e82c-249.jpg | 21 | 0 | 0 | 0 | A |
| a4e9e40d-256.jpg | 3 | 0 | 0 | 0 | A |
| a4e9e40d-256.jpg | 4 | A | A | A | 0 |
| 4960af30-167.jpg | 10 | C | C | C | B |
| 4960af30-167.jpg | 12 | A | A | A | B |
| 8eb089d6-26.jpg | 12 | B | B | B | A |
| 8eb089d6-26.jpg | 21 | A | A | A | 0 |
| 8eb089d6-26.jpg | 22 | A | A | A | 0 |

### boundary_gpt 对, softmax+ord2seq 都错 (n=23)

| image_id | roi_id | ic4 | sm_pred | ord_pred | bd_pred |
| :--- | :--- | :--- | :--- | :--- | :--- |
| f0daeb4d-28.jpg | 0 | B | A | A | B |
| f0daeb4d-28.jpg | 14 | A | 0 | 0 | A |
| 03e5a7c5-115.jpg | 10 | 0 | A | A | 0 |
| 03e5a7c5-115.jpg | 26 | 0 | A | A | 0 |
| 1131e1c7-78.jpg | 11 | A | B | B | A |
| 1355aad5-244.jpg | 6 | 0 | A | A | 0 |
| 2eca70d7-99.jpg | 9 | A | B | B | A |
| 3c160c9c-50.jpg | 6 | 0 | A | A | 0 |
| 4fc6ca9f-anonymous_003-008-762-01_1729680417375_Mandibular_View.jpg | 7 | A | 0 | 0 | A |
| 4fc6ca9f-anonymous_003-008-762-01_1729680417375_Mandibular_View.jpg | 8 | A | 0 | 0 | A |
| 4fc6ca9f-anonymous_003-008-762-01_1729680417375_Mandibular_View.jpg | 10 | A | 0 | 0 | A |
| 4fc6ca9f-anonymous_003-008-762-01_1729680417375_Mandibular_View.jpg | 13 | A | 0 | 0 | A |
| 82136b73-252.jpg | 6 | 0 | A | A | 0 |
| 82136b73-252.jpg | 17 | 0 | A | A | 0 |
| 88e305ce-74.jpg | 9 | A | 0 | 0 | A |
| 9d58e82c-249.jpg | 6 | 0 | A | A | 0 |
| 9d58e82c-249.jpg | 20 | 0 | A | A | 0 |
| a4e9e40d-256.jpg | 2 | A | 0 | 0 | A |
| b44667f5-110.jpg | 2 | A | B | 0 | A |
| e91906bc-7.jpg | 1 | A | 0 | 0 | A |
| 02e9f464-193.jpg | 6 | B | 0 | A | B |
| 02e9f464-193.jpg | 13 | A | 0 | 0 | A |
| 245f172f-21.jpg | 16 | A | 0 | 0 | A |

### softmax+ord2seq+boundary_gpt 三者都错 (n=95)

| image_id | roi_id | ic4 | sm_pred | ord_pred | bd_pred |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0b700680-102.jpg | 4 | A | 0 | 0 | 0 |
| 0b700680-102.jpg | 5 | A | 0 | 0 | 0 |
| 2bfaa34c-259.jpg | 7 | 0 | A | A | A |
| 2bfaa34c-259.jpg | 9 | A | 0 | 0 | 0 |
| 5ff09f48-anonymous_003_007_366_01_1728116583995_Maxillary_Occlusal_View.jpg | 1 | C | B | B | B |
| 5ff09f48-anonymous_003_007_366_01_1728116583995_Maxillary_Occlusal_View.jpg | 2 | C | B | B | B |
| f0daeb4d-28.jpg | 10 | B | A | A | A |
| f0daeb4d-28.jpg | 12 | B | A | A | A |
| f0daeb4d-28.jpg | 18 | 0 | A | A | A |
| f887a525-273.jpg | 14 | A | 0 | 0 | 0 |
| f9a8221b-194.jpg | 8 | 0 | A | A | A |
| 03e5a7c5-115.jpg | 11 | 0 | A | A | A |
| 03e5a7c5-115.jpg | 12 | 0 | A | A | A |
| 03e5a7c5-115.jpg | 15 | B | 0 | 0 | 0 |
| 03e5a7c5-115.jpg | 23 | 0 | A | A | A |
| 03e5a7c5-115.jpg | 25 | 0 | A | A | A |
| 03e5a7c5-115.jpg | 27 | 0 | A | A | A |
| 054ab6ae-3.jpg | 1 | A | B | B | B |
| 1131e1c7-78.jpg | 3 | 0 | A | A | A |
| 1131e1c7-78.jpg | 10 | 0 | A | A | A |
| 1355aad5-244.jpg | 0 | 0 | A | A | A |
| 27e4fdf6-235.jpg | 1 | A | B | B | B |
| 27e4fdf6-235.jpg | 3 | 0 | A | A | A |
| 2824f032-264.jpg | 3 | B | A | A | A |
| 2eca70d7-99.jpg | 2 | B | A | A | A |
| 48aa0fe4-183.jpg | 0 | B | C | C | C |
| 4fc6ca9f-anonymous_003-008-762-01_1729680417375_Mandibular_View.jpg | 2 | B | C | C | C |
| 52e40dc5-120.jpg | 4 | A | 0 | 0 | 0 |
| 52e40dc5-120.jpg | 8 | 0 | A | A | A |
| 52e40dc5-120.jpg | 16 | B | A | A | A |
| 52e40dc5-120.jpg | 17 | B | A | A | A |
| 52e40dc5-120.jpg | 30 | C | B | B | B |
| 6a4794c8-126.jpg | 0 | A | 0 | 0 | 0 |
| 6a4794c8-126.jpg | 2 | A | 0 | 0 | 0 |
| 6a4794c8-126.jpg | 9 | A | 0 | 0 | 0 |
| 76e2759f-128.jpg | 0 | C | B | B | B |
| 76e2759f-128.jpg | 3 | A | 0 | 0 | 0 |
| 82136b73-252.jpg | 0 | 0 | A | A | A |
| 82136b73-252.jpg | 1 | 0 | A | A | A |
| 82136b73-252.jpg | 2 | 0 | A | A | A |

## TEST 集

### boundary_gpt 相邻错级 (n=168)

| image_id | roi_id | ic4 | sm_pred | ord_pred | bd_pred |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1ab9d8be-30.jpg | 1 | B | B | A | A |
| 1ab9d8be-30.jpg | 6 | A | 0 | 0 | 0 |
| 1ab9d8be-30.jpg | 7 | A | A | 0 | 0 |
| 1ab9d8be-30.jpg | 9 | A | 0 | A | 0 |
| 3110e5d1-155.jpg | 1 | B | C | C | C |
| 536db3ad-17.jpg | 1 | B | B | A | A |
| 536db3ad-17.jpg | 3 | A | B | B | B |
| 536db3ad-17.jpg | 8 | B | B | A | A |
| 536db3ad-17.jpg | 11 | C | B | B | B |
| 536db3ad-17.jpg | 12 | A | 0 | 0 | 0 |
| 536db3ad-17.jpg | 19 | A | 0 | A | 0 |
| 9426ba08-43.jpg | 1 | 0 | A | A | A |
| 9426ba08-43.jpg | 4 | 0 | A | A | A |
| 9426ba08-43.jpg | 5 | 0 | A | A | A |
| b32acfa7-258.jpg | 0 | 0 | A | A | A |
| b32acfa7-258.jpg | 3 | A | 0 | 0 | 0 |
| b32acfa7-258.jpg | 6 | 0 | A | 0 | A |
| b32acfa7-258.jpg | 12 | 0 | A | A | A |
| b32acfa7-258.jpg | 14 | A | A | A | 0 |
| ba17712f-87.jpg | 2 | A | 0 | 0 | 0 |
| ba17712f-87.jpg | 4 | 0 | 0 | A | A |
| ba17712f-87.jpg | 18 | 0 | 0 | 0 | A |
| ba17712f-87.jpg | 19 | 0 | A | A | A |
| bc030429-65.jpg | 0 | A | 0 | 0 | 0 |
| bc030429-65.jpg | 3 | 0 | A | A | A |
| c636df10-156.jpg | 5 | 0 | 0 | A | A |
| c636df10-156.jpg | 8 | 0 | A | A | A |
| c636df10-156.jpg | 14 | A | 0 | 0 | 0 |
| c636df10-156.jpg | 16 | A | A | A | 0 |
| c636df10-156.jpg | 17 | A | A | A | 0 |
| c636df10-156.jpg | 18 | A | 0 | A | 0 |
| dc7d5db7-anonymous_003-007-1168-01_1732686410487_Maxillary_Occlusal_View.jpg | 1 | C | B | B | B |
| dc7d5db7-anonymous_003-007-1168-01_1732686410487_Maxillary_Occlusal_View.jpg | 2 | C | B | B | B |
| dc7d5db7-anonymous_003-007-1168-01_1732686410487_Maxillary_Occlusal_View.jpg | 3 | C | B | C | B |
| e958bf9f-107.jpg | 2 | A | 0 | C | B |
| e958bf9f-107.jpg | 4 | A | 0 | 0 | 0 |
| e958bf9f-107.jpg | 7 | 0 | A | 0 | A |
| e958bf9f-107.jpg | 13 | B | A | 0 | A |
| e958bf9f-107.jpg | 14 | B | A | 0 | A |
| e958bf9f-107.jpg | 15 | B | 0 | 0 | A |

### boundary_gpt 跨两级及以上错误 (n=9)

| image_id | roi_id | ic4 | sm_pred | ord_pred | bd_pred |
| :--- | :--- | :--- | :--- | :--- | :--- |
| e958bf9f-107.jpg | 9 | B | 0 | 0 | 0 |
| e958bf9f-107.jpg | 10 | B | A | 0 | 0 |
| e958bf9f-107.jpg | 11 | B | A | 0 | 0 |
| e958bf9f-107.jpg | 12 | B | A | 0 | 0 |
| 0ceb614e-61.jpg | 10 | B | 0 | 0 | 0 |
| 0ceb614e-61.jpg | 12 | B | 0 | 0 | 0 |
| 0ceb614e-61.jpg | 13 | B | A | 0 | 0 |
| a00bad3c-96.jpg | 19 | C | C | B | A |
| f0411d91-284.jpg | 11 | C | 0 | 0 | 0 |

### boundary_gpt 错, softmax+ord2seq 都对 (n=19)

| image_id | roi_id | ic4 | sm_pred | ord_pred | bd_pred |
| :--- | :--- | :--- | :--- | :--- | :--- |
| b32acfa7-258.jpg | 14 | A | A | A | 0 |
| ba17712f-87.jpg | 18 | 0 | 0 | 0 | A |
| c636df10-156.jpg | 16 | A | A | A | 0 |
| c636df10-156.jpg | 17 | A | A | A | 0 |
| e958bf9f-107.jpg | 17 | B | B | B | A |
| a00bad3c-96.jpg | 14 | 0 | 0 | 0 | A |
| f0411d91-284.jpg | 0 | B | B | B | A |
| 04da7e02-166.jpg | 0 | C | C | C | B |
| 15b3a4f0-282.jpg | 7 | 0 | 0 | 0 | A |
| 15b3a4f0-282.jpg | 8 | 0 | 0 | 0 | A |
| 3c6f6940-104.jpg | 5 | 0 | 0 | 0 | A |
| 3c6f6940-104.jpg | 10 | 0 | 0 | 0 | A |
| 6abd608b-234.jpg | 4 | 0 | 0 | 0 | A |
| 6abd608b-234.jpg | 10 | A | A | A | B |
| 7375bbfc-45.jpg | 5 | A | A | A | 0 |
| 85b9c68d-98.jpg | 2 | A | A | A | B |
| 85b9c68d-98.jpg | 11 | 0 | 0 | 0 | A |
| a94e3549-13.jpg | 0 | A | A | A | 0 |
| e5ba03d3-75.jpg | 4 | B | B | B | A |

### boundary_gpt 对, softmax+ord2seq 都错 (n=29)

| image_id | roi_id | ic4 | sm_pred | ord_pred | bd_pred |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 536db3ad-17.jpg | 13 | A | 0 | 0 | A |
| b32acfa7-258.jpg | 7 | 0 | A | A | 0 |
| ba17712f-87.jpg | 6 | 0 | A | A | 0 |
| ba17712f-87.jpg | 23 | 0 | A | A | 0 |
| c636df10-156.jpg | 10 | 0 | A | A | 0 |
| c636df10-156.jpg | 11 | 0 | A | A | 0 |
| e958bf9f-107.jpg | 8 | A | 0 | 0 | A |
| e958bf9f-107.jpg | 23 | 0 | A | A | 0 |
| a00bad3c-96.jpg | 7 | A | 0 | 0 | A |
| a00bad3c-96.jpg | 23 | A | 0 | 0 | A |
| a014d2cc-171.jpg | 10 | A | B | B | A |
| b3216926-anonymous_003_103_384_01_1728193820992_Maxillary_Occlusal_View.jpg | 6 | C | B | A | C |
| f0411d91-284.jpg | 7 | 0 | A | A | 0 |
| 267fc608-anonymous_003-008-1133-01_1732530707494_Mandibular_View.jpg | 3 | B | C | C | B |
| 267fc608-anonymous_003-008-1133-01_1732530707494_Mandibular_View.jpg | 5 | B | A | A | B |
| 3a073fb2-283.jpg | 0 | 0 | A | A | 0 |
| 3ab2bfb1-172.jpg | 0 | B | C | C | B |
| 3ab2bfb1-172.jpg | 2 | A | B | B | A |
| 3ab2bfb1-172.jpg | 8 | 0 | A | A | 0 |
| 3bd5f73c-34.jpg | 11 | A | 0 | 0 | A |
| 5a1e4a1b-269.jpg | 6 | A | 0 | 0 | A |
| 6abd608b-234.jpg | 6 | A | 0 | 0 | A |
| 7375bbfc-45.jpg | 4 | 0 | A | A | 0 |
| 797b1f27-anonymous_003_007_392_01_1728361981214_Mandibular_View.jpg | 9 | A | 0 | 0 | A |
| 999706d9-108.jpg | 7 | 0 | A | B | 0 |
| 999706d9-108.jpg | 23 | 0 | A | A | 0 |
| a94e3549-13.jpg | 3 | A | 0 | 0 | A |
| e5ba03d3-75.jpg | 14 | A | B | B | A |
| f23e51b4-209.jpg | 13 | 0 | A | A | 0 |

### softmax+ord2seq+boundary_gpt 三者都错 (n=96)

| image_id | roi_id | ic4 | sm_pred | ord_pred | bd_pred |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1ab9d8be-30.jpg | 6 | A | 0 | 0 | 0 |
| 3110e5d1-155.jpg | 1 | B | C | C | C |
| 536db3ad-17.jpg | 3 | A | B | B | B |
| 536db3ad-17.jpg | 11 | C | B | B | B |
| 536db3ad-17.jpg | 12 | A | 0 | 0 | 0 |
| 9426ba08-43.jpg | 1 | 0 | A | A | A |
| 9426ba08-43.jpg | 4 | 0 | A | A | A |
| 9426ba08-43.jpg | 5 | 0 | A | A | A |
| b32acfa7-258.jpg | 0 | 0 | A | A | A |
| b32acfa7-258.jpg | 3 | A | 0 | 0 | 0 |
| b32acfa7-258.jpg | 12 | 0 | A | A | A |
| ba17712f-87.jpg | 2 | A | 0 | 0 | 0 |
| ba17712f-87.jpg | 19 | 0 | A | A | A |
| bc030429-65.jpg | 0 | A | 0 | 0 | 0 |
| bc030429-65.jpg | 3 | 0 | A | A | A |
| c636df10-156.jpg | 8 | 0 | A | A | A |
| c636df10-156.jpg | 14 | A | 0 | 0 | 0 |
| dc7d5db7-anonymous_003-007-1168-01_1732686410487_Maxillary_Occlusal_View.jpg | 1 | C | B | B | B |
| dc7d5db7-anonymous_003-007-1168-01_1732686410487_Maxillary_Occlusal_View.jpg | 2 | C | B | B | B |
| e958bf9f-107.jpg | 2 | A | 0 | C | B |
| e958bf9f-107.jpg | 4 | A | 0 | 0 | 0 |
| e958bf9f-107.jpg | 9 | B | 0 | 0 | 0 |
| e958bf9f-107.jpg | 10 | B | A | 0 | 0 |
| e958bf9f-107.jpg | 11 | B | A | 0 | 0 |
| e958bf9f-107.jpg | 12 | B | A | 0 | 0 |
| e958bf9f-107.jpg | 13 | B | A | 0 | A |
| e958bf9f-107.jpg | 14 | B | A | 0 | A |
| e958bf9f-107.jpg | 15 | B | 0 | 0 | A |
| e958bf9f-107.jpg | 19 | B | A | A | A |
| e958bf9f-107.jpg | 24 | 0 | A | A | A |
| 0ceb614e-61.jpg | 6 | A | 0 | 0 | 0 |
| 0ceb614e-61.jpg | 10 | B | 0 | 0 | 0 |
| 0ceb614e-61.jpg | 11 | B | A | A | A |
| 0ceb614e-61.jpg | 12 | B | 0 | 0 | 0 |
| 0ceb614e-61.jpg | 13 | B | A | 0 | 0 |
| 0ceb614e-61.jpg | 16 | 0 | A | A | A |
| a00bad3c-96.jpg | 5 | A | 0 | 0 | 0 |
| a00bad3c-96.jpg | 10 | A | 0 | 0 | 0 |
| a00bad3c-96.jpg | 17 | 0 | A | A | A |
| a00bad3c-96.jpg | 20 | 0 | A | A | A |
