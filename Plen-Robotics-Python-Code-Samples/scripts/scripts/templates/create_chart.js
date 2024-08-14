// creates a chart from a  using chart.js
const labels = [
      'January',
      'February',
      'March',
    ];

    const data = {
      labels: labels,
      datasets: [{
        label: 'My first dataset',
        backgroundColor: 'rgb(255, 99, 132)',
        borderColor: 'rgb(255, 99, 132)',
        data: [0, 10, 5],
      }]
    };
    
    const config = {
      type: 'bar',
      data: data,
      options: {
        indexAxis: 'y',
      }
    }

    const myChart = new Chart(
      document.getElementById('chart'),
      config
    );
/*
d3.csv("../totals.csv")
.then(makeChart);

function makeChart(data) {
  var date = data.map(function(d) {return d.date});
  var male = data.map(function(d) {return d.male});
  var female = data.map(function(d) {return d.female});
  // Bar chart
  var chart = new Chart(document.getElementById("summary"), {
      type: 'horizontalBar',
      data: {
        labels: date,
        datasets: [
          {
            data: male
          }
        ]
      }
  });
  */