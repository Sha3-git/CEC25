const express = require('express');
const axios = require('axios');
const app = express();
const PORT = 4001;

app.use(express.json());

app.post('/predict', async (req, res) => {
  try {
    const modelInput = req.body.input;
    const response = await axios.post('http://localhost:5001/predict', {
      input: modelInput,
    });

    const prediction = response.data;
    res.json(prediction);
  } catch (error) {
    res.status(500).json({ error: 'Error processing request' });
  }
});

app.listen(PORT, () => {
  console.log(`server running on port ${PORT}`);
});
