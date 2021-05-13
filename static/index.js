function classify() {
    var resultDiv = document.getElementById("result");
    resultDiv.innerHTML = "<p class='wait'>Loading...</p>";
    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function () {
        if (this.readyState === 4) {
            var resultJSON = JSON.parse(xhr.responseText);
            if (this.status === 200) {
                resultDiv.innerHTML = "<p class='" + resultJSON.sentiment.toLowerCase() + "'>" + resultJSON.sentiment + "\t" + Math.round((resultJSON.confidence_score + Number.EPSILON) * 10000) / 100 + "%</p>"
            } else {
                resultDiv.innerHTML = "<p class='error'>" + resultJSON.error + "</p>";
            }
        }
    }
    var text = document.getElementById("text").value;
    xhr.open("GET", "/classify?text=" + text);
    xhr.send();
}