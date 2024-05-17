import timm

model_names = timm.list_models('*hrnet*')
print(model_names)