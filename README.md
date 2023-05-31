![image](https://github.com/karnjj/sed-young-wa/assets/60351130/c1196b90-37ea-481a-b871-0ba634a58ccb)

# Sed-Young-Wa 

Sed-Young-Wa (เสร็จยังหว่า) is a service for estimating the end of the issues submitted in the Traffy fondue platform. We built a regression model `GradientBoostingRegressor` with features from the issues including
issue types, submission date, and district. And also uses scraped features: weather and holidays in Thailand. After that, We constructed a model pipeline and stored model variables on MLFlow to retrieve the model 
in production. The model results achieved a root-mean-squared error (RMSE) of 20.

# Diagram
## Training phase
<p align="center"><image src="https://github.com/karnjj/sed-young-wa/assets/60351130/c650fc14-689b-493e-b540-e2e042283baf" width=800 /></p>

## Run-time phase
<p align="center"><image src="https://github.com/karnjj/sed-young-wa/assets/60351130/2491560c-f22a-43c3-be02-1a4df099b49c" width=800 /></p>


# Appendix

1. You can view the model pipeline process in the notebook [link](https://github.com/karnjj/sed-young-wa/blob/main/research-notebook.ipynb).
2. A clip of the final report is shown [here](https://www.youtube.com/watch?v=mdGsco10JPI).

# Authors

- [@NonKhuna](https://github.com/NonKhuna) [Data science]: Create a model and preprocess Data.
- [@Karn](https://github.com/karnjj) [Data engineering]: Model pipeline.
- [@Tantawit](https://github.com/Tantawit) [Data engineering]: Scraping Data
- [@Anon](https://github.com/Anon-136) [Data visualization]

