from habibi import PlantDiseasePredictor, load_class_names

class_names = load_class_names('class_names.txt')
predictor = PlantDiseasePredictor('best_transfer_learning_model.h5', class_names)
results = predictor.predict('tesr.png')
print(results)

# bacterial
# [{'class': 'Tomato___Bacterial_spot', 'confidence': 94.30521392822266}, {'class': 'Tomato___Septoria_leaf_spot', 'confidence': 2.18812894821167}, {'class': 'Tomato___Early_blight', 'confidence': 1.8525866270065308}]

# septorial
# [{'class': 'Tomato___Septoria_leaf_spot', 'confidence': 99.87089538574219}, {'class': 'Tomato___Early_blight', 'confidence': 0.09244897216558456}, {'class': 'Tomato___Tomato_mosaic_virus', 'confidence': 0.02295910008251667}]

# 
# [{'class': 'Tomato___Late_blight', 'confidence': 60.45942306518555}, {'class': 'Tomato___healthy', 'confidence': 39.02615737915039}, {'class': 'Tomato___Septoria_leaf_spot', 'confidence': 0.3928041458129883}]