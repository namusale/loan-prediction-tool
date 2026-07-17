/*
 The code in module works to manage information keyed in by the user,
 sends user information to the backend and then awaits for response from
 the back end and updates the results in the results section
*/
import { useState } from 'react'
import Features from './comps/featureDetails.jsx'
import Results from './comps/results.jsx'
import './App.css'

function App() {

// User input information
  const [features, setFeatures] = useState({
        cibil_score:0,
        loan_term:0,
        income_annum:0,
        education_not_graduate:"",
        loan_amount:0,
        self_employed:"",
        luxury_assets_value:0
        });
  const [results,setResults]= useState(2);

//Sending data to backend and awaits reponse
  async function getdata() {
    setResults(3)

  try {
    let response = await fetch("http://127.0.0.1:8000/predict/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(features)
    });

    if (!response.ok) {
      setResults(4)
      let errorJson = await response.json();
      alert("FASTAPI ERROR DETECTED:\n" + JSON.stringify(errorJson, null, 2));
      setResults(4)
      return;
    }

    const res = await response.json()
    alert(JSON.stringify(res, null, 2))
    const output = res["output"]
    setResults(output)
    alert(output)

  } catch (error) {
    setResults(4)
    alert("Network/Fetch Error: " + error);
  }
}
  return (
    <>
    <h1>Loan Approval Prediction Tool</h1>

    <div className = "showData">
      <section className="Features">
      {/*Sharing data with feature component*/}
        <Features
            features = {features}
            setFeatures = {setFeatures}
            getdata ={getdata}
        />
      </section>

      {/*Sharing data with Results component*/}
      <section className="Results">
        <Results
          results = {results}
          setResults = {setResults}
            />
        </section>
      </div>
    </>
  )
}
export default App
