// This module provides for user to input personal data for loan approval evaluation
export default function Features({features,setFeatures,getdata}){

const localFeatures = Object.keys(features) // listing feature keys

// Input sanity checking function
function sanityCheck(features){

    if (isNaN(Number(features.cibil_score)) || Number(features.cibil_score) < 0){
    alert("The CIBIL score must be a number greater than Zero")
    }

     if (isNaN(Number(features.loan_term)) || Number(features.loan_term) < 0){
    alert("The Loan Term must be greater than zero")
    }

    if (isNaN(Number(features.income_annum)) || Number(features.income_annum) < 0){
    alert("The Income Annum must not be less than zero")
    }

//     if (((features.education_not_graduate.toLowerCase().trim() != "no") || (features.education_not_graduate.toLowerCase().trim() != "yes")) &&(features.education_not_graduate.toLowerCase().trim().length>=2)){
//     alert("Please enter Yes or No ")
//     }
// //
//     if ((features.self_employed.toLowerCase() != "no") || (features.self_employed.toLowerCase() != "yes")){
//     alert("Please enter Yes or No ")
//     }

        if (isNaN(Number(features.loan_amount)) || Number(features.loan_amount) < 0){
    alert("The Loan Amount must be greater than zero")
    }

        if (isNaN(Number(features.luxury_assets_value)) || Number(features.luxury_assets_value) < 0){
    alert("The Luxury Assets Value must not be less than zero")
    }


}

// Object for user friendly feture names
const dictFeatures = new Map([
                            [localFeatures[0],'CIBIL Score  ' ],
                           [localFeatures[1],'Loan Term  '],
                        [localFeatures[2],'Income Annum  '],
                        [localFeatures[3],'Do you hold a Graduate Degree?  '],
                       [localFeatures[4],'Loan Amount  '],
                       [localFeatures[5],'Are You Self Employed?  '],
                        [localFeatures[6],'Luxury Assets Value  ' ]
                        ]);

// Handling Inputs
function handleChange(e){
    const name = e.target.name
    const value = e.target.value
    setFeatures(values=>({...values,[name]:value}))
}

// Feature frontend Display
const localFeature = localFeatures.map((feature)=>{
            return(<>
                <ul>
                    <li>
                        <label key ={feature}>
                          {dictFeatures.get(feature)}

                            <input className="inputs"
                                type = 'text'
                                name = {feature}
                                value ={features[feature] || ""}
                                onChange={handleChange}
                                />

                        </label>
                     </li>
                </ul>
                    </>)})

//submitting data to backend
function handleSubmit(e){
    e.preventDefault()
    getdata()
    }

return(
<>
    <h2>Personal Details</h2>
    <div>Please enter your personal information below:</div>
    <form onSubmit = {handleSubmit}>
        <div>{localFeature}</div>
        <input className = "submit" type ='submit'/>
      </form>
    <div>{sanityCheck(features)}</div>
</>
)}

