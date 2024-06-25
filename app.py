from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('modelo_tesla.pyl')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        open = float(request.form['open'])
        close = float(request.form['close'])
        volume = float(request.form['volume'])
        avg_vol_20d = float(request.form['avg_vol_20d'])

        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[open,close,volume,avg_vol_20d]], columns=['open', 'close','volume','avg_vol_20d'])
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Realizar predicciones
        prediction = model.predict(data_df)
        app.logger.debug(f'Predicción: {prediction[0]}')
        

        # Devolver las predicciones como respuesta JSON
        return jsonify({'Precio': prediction[0]})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

