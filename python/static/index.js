root = ""

var colormap = chroma.scale(['purple', 'blue', 'green', 'yellow', 'red', 'black']);
$(document).ready(function () {
    $("#powerform").hide();
    var slider_value = 0;
    $("#windslider").roundSlider({
        min: 0,
        max: 359,
        value: 90,
        editableTooltip: false,
        showTooltip: false,
        change: function (e) {
            console.log(e.value);
            slider_value = e.value - 90;
            if (slider_value < 0) {
                slider_value += 360;
            }
        }
    });
    $("#profileform").submit(function (event) {
        event.preventDefault();
        $("#profileresult").html('<div class="spinner-border m-5" role="status"> <span class="sr-only">Loading...</span> </div>');
        $("#profilemap").html('<div class="spinner-border m-5" role="status"> <span class="sr-only">Loading...</span> </div>');
        url = root + "/api/profile" +
            "?id=" + String($("#segmentid").val()) +
            "&use_ign=" + String(1.0 * ($("#correctalt").is(":checked"))) +
            "&use_gmaps=" + String(1.0 * ($("#snaptoroads").is(":checked"))) +
            "&pen=" + String($("#pen").val());
        $.getJSON(url, function (data) {
            $("#profilename").html(data["segment_name"]);
            var d1 = []
            for (var i = 0; i < data["segment_points"].length; i++) {
                var lastPoint = [Number.parseFloat(data["segment_points"][i]["distance"]), Number.parseFloat(data["segment_points"][i]["alt"])];
                d1.push(lastPoint);
            }
            var plotdata = [
                {
                    data: d1,
                    lines: { show: true, fill: false },
                    points: { show: true }
                }];
            options = {
                legend: {
                    show: false
                },
                xaxis: {

                }
            }
            for (var i = 0; i < data["profile_points"].length - 1; i++) {
                d3 = []
                d3.push([Number.parseFloat(data["profile_points"][i]["distance"]), 0]);
                d3.push([Number.parseFloat(data["profile_points"][i]["distance"]), Number.parseFloat(data["profile_points"][i]["alt"])]);
                d3.push([Number.parseFloat(data["profile_points"][i + 1]["distance"]), Number.parseFloat(data["profile_points"][i + 1]["alt"])]);
                d3.push([Number.parseFloat(data["profile_points"][i + 1]["distance"]), 0]);
                color = Number.parseFloat(data["profile_intervals"][i]["grade"]) / 20 + 0.5;
                c = 0.5 * (Number.parseFloat(data["profile_points"][i]["distance"]) + Number.parseFloat(data["profile_points"][i + 1]["distance"]))
                label = Number.parseFloat(data["profile_intervals"][i]["grade"]).toFixed(1) + "%";
                console.log(c);
                console.log(label);
                plotdata.push({
                    data: d3,
                    lines: { show: true, fill: true },
                    label: label,
                    points: { show: false },
                    color: colormap(color)
                });
            }
            $.plot("#profileresult", plotdata, options);
            var d4 = []
            var map_plotdata = []
            lastPoint = null;
            for (var i = 0; i < data["profile_points"].length - 1; i++) {
                d4 = []
                if(lastPoint) {
                    d4.push(lastPoint);
                }
                color = Number.parseFloat(data["profile_intervals"][i]["grade"]) / 20 + 0.5;
                for (var j = 0; j < data["segment_points"].length; j++) {
                    if (data["segment_points"][j]["distance"] >= data["profile_points"][i]["distance"] &&
                        data["segment_points"][j]["distance"] < data["profile_points"][i + 1]["distance"]) {
                        lat = data["segment_points"][j]["lat"];
                        lng = data["segment_points"][j]["lng"];
                        lat = Math.PI * lat / 180;
                        lng = Math.PI * lng / 180;
                        // adjust position by radians
                        lat -= 1.570795765134; // subtract 90 degrees (in radians)

                        // and switch z and y
                        xPos = lng * Math.cos(lat);
                        yPos = lat
                        d4.push([xPos, yPos]);
                        lastPoint = [xPos, yPos];
                    }
                }
                map_plotdata.push({
                    data: d4,
                    lines: { show: true, fill: false },
                    points: { show: true },
                    color: colormap(color)
                });
            }
            // start and finish
            d5 = []
            lat = data["segment_points"][0]["lat"];
            lng = data["segment_points"][0]["lng"];
            lat = Math.PI * lat / 180;
            lng = Math.PI * lng / 180;
            lat -= 1.570795765134; // subtract 90 degrees (in radians)
            xPos = lng * Math.cos(lat);
            yPos = lat
            d5.push([xPos, yPos]);
            map_plotdata.push({
                data: d5,
                lines: { show: true, fill: false },
                points: { show: true , radius: 10},
                color: "green"
            });

            d6 = []
            lat = data["segment_points"][data["segment_points"].length - 1]["lat"];
            lng = data["segment_points"][data["segment_points"].length - 1]["lng"];
            lat = Math.PI * lat / 180;
            lng = Math.PI * lng / 180;
            lat -= 1.570795765134; // subtract 90 degrees (in radians)
            xPos = lng * Math.cos(lat);
            yPos = lat
            d6.push([xPos, yPos]);
            map_plotdata.push({
                data: d6,
                lines: { show: true, fill: false },
                points: { show: true , radius: 10},
                color: "red"
            });




            options = { legend: { show: false }, xaxis: { show: false }, yaxis: { show: false } };
            $("#profilemap").html("");
            $.plot("#profilemap", map_plotdata, options);
            $("#powerform").show()
        })
            .fail(function () {
                $("#profilemap").html("Error getting profile");
                $("#profileresult").html("");
            });
    });
    $("#powerform").submit(function (event) {
        event.preventDefault();
        $("#powerresult").html('<div class="spinner-border m-5" role="status"> <span class="sr-only">Loading...</span> </div>');
        url = root + "/api/optimal_power" + "?mass=" + String($("#mass").val()) + "&power=" + String($("#power").val()) +
            "&id=" + String($("#segmentid").val()) + "&Cda=" + String($("#cda").val()) +
            "&Cr=" + String($("#cr").val()) +
            "&wind_speed=" + String($("#windspeed").val()) +
            "&wind_direction=" + String(slider_value) +
            "&temp=" + String($("#temp").val()) +
            "&use_ign=" + String(1.0 * ($("#correctalt").is(":checked"))) +
            "&constant_power=" + String(1.0 * ($("#constant_power").is(":checked"))) +
            "&use_gmaps=" + String(1.0 * ($("#snaptoroads").is(":checked"))) +
            "&pen=" + String($("#pen").val()) +
            "&drivetrain_efficiency=" + String($("#efficiency").val());
        $.getJSON(url, function (data) {
            var items = [];
            html = '<table><thead class="thead-dark m-5"> <tr>' +
                '<th scope="col">Distance (km)</th>' +
                '<th scope="col">Length (km)</th> <th scope="col">Elevation (m)</th>' +
                '<th scope="col">Time</th>' +
                '<th scope="col">Grade (%)</th>' +
                '<th scope="col">Power (W)</th>' +
                '<th scope="col">Speed (km/h)</th>' +
                '<th scope="col">Average (km/h)</th>' +
                '<th scope="col">Headwind (km/h)</th>' +
                '</tr> </thead>';
            $.each(data["intervals"], function (key, val) {
                var date = new Date(0);
                date.setSeconds(val['time']); // specify value for SECONDS here
                var timeString = date.toISOString().substr(11, 8);
                items.push('<tr><th scope="row">' + (Number.parseFloat(val['distance']) / 1000).toFixed(1) +
                    ' </th><td>' + (Number.parseFloat(val['length']) / 1000).toFixed(1) +
                    ' </td><td>' + Number.parseFloat(val['total_elevation']).toFixed(1) +
                    ' </td><td>' + timeString +
                    ' </td><td>' + Number.parseFloat(val['grade']).toFixed(1) +
                    ' </td><td>' + Number.parseFloat(val['power']).toFixed(1) +
                    ' </td><td>' + Number.parseFloat(val['speed_kmh']).toFixed(1) +
                    ' </td><td>' + Number.parseFloat(val['average_kmh']).toFixed(1) +
                    ' </td><td>' + (Number.parseFloat(val['headwind'])*3.6).toFixed(1) +
                    '</td></tr>');
            });
            $("#powerresult").html(

                $("<table/>", {
                    "class": "table",
                    html: html + items.join("") + "</table>"
                }))
        });
    });
});
