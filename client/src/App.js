import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import Predict from './pages/Predict';

import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Predict />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
/*
This code was created with the help AI
*/