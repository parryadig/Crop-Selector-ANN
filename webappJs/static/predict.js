let model;
(async function() {
    model = await tf.loadLayersModel("http://0.0.0.0:81/tfjs-models/crop_pred_ann/model.json");
})();

let arrayMaxIndex = function(array) {
    return array.indexOf(Math.max.apply(null, array));
  };

let labels = ['Adzuki Beans', 'Black gram', 'Chickpea', 'Coconut', 'Coffee',
                'Cotton', 'Ground Nut', 'Jute', 'Kidney Beans', 'Lentil',
                'Moth Beans', 'Mung Bean', 'Peas', 'Pigeon Peas', 'Rubber',
                'Sugarcane', 'Tea', 'Tobacco', 'apple', 'banana', 'grapes',
                'maize', 'mango', 'millet', 'muskmelon', 'orange', 'papaya',
                'pomegranate', 'rice', 'watermelon', 'wheat']

$("#predict-btn").click(async function () {
    let a = parseInt(document.getElementById("a").value);
    let b = parseInt(document.getElementById("b").value);
    let c = parseInt(document.getElementById("c").value);
    let d = parseInt(document.getElementById("d").value);
    console.log(a, b, c, d);
    let tensor = tf.tensor([[a, b, c, d]]);

    let predictions_raw = await model.predict(tensor).data();
    console.log("Prediction complete.");
    let prediction = labels[arrayMaxIndex(predictions_raw)];
    console.log(prediction);

    $("#prediction-list").empty();
    $("#prediction-list").append(`<h2>${prediction}<h2>`);
    
});