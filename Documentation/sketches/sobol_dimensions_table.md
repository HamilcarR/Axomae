|Dimensions|Purpose	|Tier|Rationale|
|---|---|---|---|
| 0, 1|	Pixel Jitter (x, y)|	S	|The foundation of the entire pixel sample. Must be first.|
| 2, 3|	Lens Sample (u, v)|	A	|Determines initial ray origin for DoF. High impact.|
| 4	|Time Sample|	A|	Determines initial ray time for motion blur. High impact.|
| 5, 6|	1st Bounce BSDF Sample|	A	|Critical first direction choice.|
| 7	|1st Bounce Light Choice|	A|	Which light to sample for direct illumination.|
| 8, 9|	1st Bounce Light Position Sample|	A|	Position on the chosen light.|
| 10|	1st Bounce Russian Roulette|	B|	Decides if we continue to the 2nd bounce.|
| 11, 12|	2nd Bounce BSDF Sample|	B|	Less important than 1st bounce.|
