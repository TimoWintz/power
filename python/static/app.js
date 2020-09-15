root = ""
var colormap = chroma.scale(['green', 'yellow', 'red', 'black']);
$(document).ready(function() {
        $("#profileform").submit(function( event ) {
            event.preventDefault();
            $("#profileresult").html('<div class="spinner-border m-5" role="status"> <span class="sr-only">Loading...</span> </div>');
            url = root + "/api/profile" +
                "?id=" + String($("#segmentid").val()) +
                "&pen=" + String($("#pen").val());
            $.getJSON(url, function( data ) {
                var d1 = []
                for (var i = 0; i < data["segment_points"].length; i++) {
                    d1.push([Number.parseFloat(data["segment_points"][i]["distance"]), Number.parseFloat(data["segment_points"][i]["alt"])]);
                }
                var plotdata = [
                    {
                        data: d1,
                        lines: { show: true, fill: false},
                        points: {show: true}
                    }];
                options = {
                    legend: {
                        show: false
                    },
                    xaxis: {

                    }
                }
                for (var i = 0; i < data["profile_points"].length-1; i++) {
                    d3 = []
                    d3.push([Number.parseFloat(data["profile_points"][i]["distance"]), 0]);
                    d3.push([Number.parseFloat(data["profile_points"][i]["distance"]), Number.parseFloat(data["profile_points"][i]["alt"])]);
                    d3.push([Number.parseFloat(data["profile_points"][i+1]["distance"]), Number.parseFloat(data["profile_points"][i+1]["alt"])]);
                    d3.push([Number.parseFloat(data["profile_points"][i+1]["distance"]), 0]);
                    color = Number.parseFloat(data["profile_intervals"][i]["grade"]) / 20 + 0.5;
                    c = 0.5*(Number.parseFloat(data["profile_points"][i]["distance"]) + Number.parseFloat(data["profile_points"][i+1]["distance"]))
                    label = Number.parseFloat(data["profile_intervals"][i]["grade"]).toFixed(1) + "%";
                    console.log(c);
                    console.log(label);
                    plotdata.push({
                        data: d3,
                        lines: { show: true, fill: true},
                        label: label,
                        points: {show: false},
                        color: colormap(color)
                    });
                }
            $.plot("#profileresult", plotdata, options);
            });
        });
        $("#powerform").submit(function( event ) {
            event.preventDefault();
            $("#powerresult").html('<div class="spinner-border m-5" role="status"> <span class="sr-only">Loading...</span> </div>');
            url = root + "/api/optimal_power" + "?mass=" + String($("#mass").val()) + "&power=" + String($("#power").val()) +
                "&id=" + String($("#segmentid").val()) + "&Cda=" + String($("#cda").val()) +
                "&Cr=" + String($("#cr").val()) +
                "&pen=" + String($("#pen").val()) +
                "&drivetrain_efficiency=" + String($("#efficiency").val()) ;
            $.getJSON(url, function( data ) {
              var items = [];
              html = '<thead class="thead-dark m-5"> <tr> <th scope="col">#</th>'+
                    '<th scope="col">Distance (km)</th> <th scope="col">Elevation (m)</th>'+
                    '<th scope="col">Time</th>'+
                    '<th scope="col">Grade (%)</th>' +
                    '<th scope="col">Power (W)</th>' +
                    '<th scope="col">Speed (km/h)</th>' +
                    '<th scope="col">Average (km/h)</th>' +
                    '</tr> </thead>';
              $.each( data, function( key, val ) {
              var date = new Date(0);
              date.setSeconds(val['time']); // specify value for SECONDS here
              var timeString = date.toISOString().substr(11, 8);
                items.push( '<tr><th scope="row">' + key +
                    ' </th><td>' + (Number.parseFloat(val['distance'])/1000).toFixed(1) +
                    ' </th><td>' + Number.parseFloat(val['cum_elevation']).toFixed(1) +
                    ' </th><td>' + timeString +
                    ' </th><td>' + Number.parseFloat(val['grade']).toFixed(1) +
                    ' </th><td>' + Number.parseFloat(val['power']).toFixed(1) +
                    ' </th><td>' + Number.parseFloat(val['speed_kmh']).toFixed(1) +
                    ' </th><td>' + Number.parseFloat(val['average_kmh']).toFixed(1) +
                '</td></tr>' );
              });
              $("#powerresult").html(
             
              $( "<table/>", {
                "class": "table",
                html: html + items.join( "" )
              }))
            });
        });
        $.ajax({
                    url: root + "/api/auth"
        }).then(function(data) {
            if (data.auth == false) {
                window.location.replace(root);
            } else {
               $('#name').append("Logged in as: ", data.firstname);
            }
        });
});
