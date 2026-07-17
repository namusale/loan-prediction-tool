/* The module works to display the results from the backend
provided by the app.jsx module. */

export default function Results({results,setResults}){

// Displaying results intermediate status and final outcome
function outcomes(results){
    const outcome = results;
    if (outcome == 0){
        return( <div> Congraturations!! Your Loan has been approved.</div>)}
    else if (outcome == 1){
        return(<div>Im sorry. Your Loan application has not been approved.</div>)}
    else if (outcome == 2){
        return(
                <>
                <div style ={{textIndent: "50px",fontWeight:"bold"}}>Loan Status: </div>
                <div style ={{textIndent: "120px"}}>Awaiting submission </div>

                </>)}
    else if (outcome == 3){
        return(<div>Waiting Results......</div>)}
    else if (outcome == 4){
        return(<div>Please try again.</div>)}

    }

return(
    <>
         <h2>Results</h2>
        <div>{outcomes(results)}</div>
    </>
)
};

