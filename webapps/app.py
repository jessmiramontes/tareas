from flask import Flask, request, render_template
import joblib
import pandas as pd
import logging

logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
app = Flask(__name__)

model = joblib.load('optimized_knn_pipeline_model.joblib')
class_dict = {
    "0": "Insufficient_Weight",
    "1": "Normal_Weight",
    "2": "Obesity_Type_I",
    "3": "Obesity_Type_II",
    "4": "Obesity_Type_III",
    "5": "Overweight_Level_I",
    "6": "Overweight_Level_II"
}

# Set up logging 
logging.basicConfig(level=logging.DEBUG)

@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get form data
            gender = request.form['gender']
            age = float(request.form['age'])
            height = float(request.form['height'])
            weight = float(request.form['weight'])
            family_history_with_overweight = request.form['has_a_family_member_suffered_or_suffers_from_overw']
            favc = request.form['do_you_eat_high_caloric_food_frequently']
            fcvc = request.form['do_you_usually_eat_vegetables_in_your_meals']
            ncp = float(request.form['how_many_main_meals_do_you_have_daily'])
            caec = request.form['do_you_eat_any_food_between_meals']
            smoke = request.form['do_you_smoke']
            ch2o = float(request.form['how_much_water_do_you_drink_daily'])
            scc = request.form['do_you_monitor_the_calories_you_eat_daily']
            faf = float(request.form['how_often_do_you_have_physical_activity'])
            tue = float(request.form['how_much_time_do_you_use_technological_devices_suc'])
            calc = request.form['how_often_do_you_drink_alcohol']
            mtrans = request.form['which_transportation_do_you_usually_use']
            my_df = pd.DataFrame([[gender, age, height, weight, family_history_with_overweight, favc, fcvc, ncp, caec, smoke, ch2o, scc, faf, tue, calc, mtrans]], columns= ["gender", "age", "height", "weight", "family_history_with_overweight", "favc", "fcvc", "ncp", "caec", "smoke", "ch2o", "scc", "faf", "tue", "calc", "mtrans"] )
            # Log the DataFrame 
            logging.debug(f"Input DataFrame: {my_df}")
            prediction = model.predict(my_df)[0]
            prediction_str = str(prediction)
            logging.debug(f"Prediction (as string): {prediction_str}")
            pred_class = class_dict[prediction_str]
            return render_template("index.html", prediction=pred_class)
        except Exception as e: 
            logging.error(f"Error during prediction: {e}") 
            return str(e), 400
    else:
        return render_template("index.html", prediction=None)
    
if __name__ == '__main__': 
    app.run(debug=True)
