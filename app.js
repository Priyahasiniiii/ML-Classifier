document.getElementById('input-form').addEventListener('submit', async function(event) {
    event.preventDefault();
  
    // Get the input values from the form
    const height = parseFloat(document.getElementById('height').value);
    const weight = parseFloat(document.getElementById('weight').value);
  
    // Make sure the values are valid
    if (isNaN(height) || isNaN(weight)) {
      alert("Please enter valid numbers for height and weight.");
      return;
    }
  
    // Load or define the ML model
    const model = await loadModel();
  
    // Use the model to classify the input values
    const prediction = model.predict(tf.tensor2d([[height, weight]])).dataSync();
  
    // Show the result in the HTML
    const resultText = prediction[0] > 0.5 ? "Class: A" : "Class: B";
    document.getElementById('classification-result').textContent = resultText;
  });
  
  // Load the pre-trained model (you can use a custom model or train one yourself)
  async function loadModel() {
    // For simplicity, we'll define a dummy model here
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [2], activation: 'sigmoid' }));
    
    // Compile the model
    model.compile({ loss: 'binaryCrossentropy', optimizer: 'adam', metrics: ['accuracy'] });
  
    // Example dataset for training (height, weight, label)
    const xs = tf.tensor2d([
      [170, 70], [180, 80], [160, 60], [155, 50], [165, 65], [175, 75]
    ]);
    const ys = tf.tensor2d([[1], [1], [0], [0], [1], [1]]);  // Class A (1) or Class B (0)
  
    // Train the model (this is simplified for demonstration)
    await model.fit(xs, ys, { epochs: 100 });
  
    return model;
  }
  