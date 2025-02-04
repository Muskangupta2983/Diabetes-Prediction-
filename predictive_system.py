import numpy as np
import pickle 

loaded_model = pickle.load(open('C:/Users/Somendra Mishra/OneDrive/Desktop/Diabetes/trained_model.sav','rb'))

input_data = (8,183,64,0,0,23.3,0.672,32)

# changing the input_data to numpy array

input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#standardize the input data
#std_data = scaler.transform(input_data_reshaped)
#print(std_data)


prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
  print("The person is not diabetic")
else:
  print("The person is diabetic")