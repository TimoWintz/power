root = ""
var profile_data = null;
var slider_value = 0;

function project_web_mercator(lng, lat) {
    lat = Math.PI * lat / 180;
    lng = Math.PI * lng / 180;
    xPos = lng + Math.PI;
    yPos = Math.log(Math.tan(Math.PI/4 + lat / 2));
    return [xPos, yPos];
}

function profile(event) {
    event.preventDefault();
    $("#profileresult").html('<div class="spinner-border m-5" role="status"> <span class="sr-only">Loading...</span> </div>');
    $("#profilemap").html('<div class="spinner-border m-5" role="status"> <span class="sr-only">Loading...</span> </div>');
    url = root + "/api/profile" +
        "?id=" + String($("#segmentid").val()) +
        "&type=" + String($("#select_type").val()) +
        "&use_ign=" + String(1.0 * ($("#correctalt").is(":checked"))) +
        "&use_gmaps=" + String(1.0 * ($("#snaptoroads").is(":checked"))) +
        "&Cda=" + String($("#cda").val()) +
        "&Cr=" + String($("#cr").val()) +
        "&mass=" + String($("#mass").val()) +
        "&wind_speed=" + String($("#windspeed").val()) +
        "&wind_direction=" + String(slider_value) +
        "&temp=" + String($("#temp").val()) +
        "&pen=" + String($("#pen").val());
    $.getJSON(url, function (data) {
        profile_data = data;
        $("#profilename").html(data["segment_name"]);
        var d1 = []
        for (var i = 0; i < data["segment_points"].length; i++) {
            var lastPoint = [Number.parseFloat(data["segment_points"][i]["distance"]/1000), Number.parseFloat(data["segment_points"][i]["alt"])];
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

                
            },
            grid:{autoHighlight:false, hoverable:true, clickable:false, axisMargin:10, borderWidth:0}
        };
        
        for (var i = 0; i < data["profile_points"].length - 1; i++) {
            d3 = []
            d3.push([Number.parseFloat(data["profile_points"][i]["distance"]/1000), 0]);
            d3.push([Number.parseFloat(data["profile_points"][i]["distance"]/1000), Number.parseFloat(data["profile_points"][i]["alt"])]);
            d3.push([Number.parseFloat(data["profile_points"][i + 1]["distance"]/1000), Number.parseFloat(data["profile_points"][i + 1]["alt"])]);
            d3.push([Number.parseFloat(data["profile_points"][i + 1]["distance"]/1000), 0]);
            color = Number.parseFloat(data["profile_intervals"][i]["grade"]) / 20 + 0.5;
            label = Number.parseFloat(data["profile_intervals"][i]["grade"]).toFixed(1) + "%";
            plotdata.push({
                data: d3,
                lines: { show: true, fill: true },
                label: label,
                points: { show: false },
                color: colormap(color)
            });
            
        }
        
        $.plot("#profileresult", plotdata, options);
        

        $("#profileresult").bind("plothover", function (event, pos, item) {
            $("#tooltip").show();
            $("#tooltip").css({top: pos.pageY - 30, left: pos.pageX});
            for (var i = 1; i < profile_data["profile_points"].length; i++) {
                if (profile_data["profile_points"][i]["distance"]/1000 > pos.x) {
                    $("#tooltip").html((profile_data["profile_intervals"][i-1]["length"]/1000).toFixed(1) + "km at " + profile_data["profile_intervals"][i-1]["grade"].toFixed(1) + "%");
                    break;
                }
                
            }
                
        });
        $("#profileresult").mouseleave(function () {  $("#tooltip").hide(); });
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
                    var xy = project_web_mercator(lng, lat);
                    d4.push(xy);
                    lastPoint = xy;
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
        var xy = project_web_mercator(lng, lat);
        
        d5.push(xy);
        map_plotdata.push({
            data: d5,
            lines: { show: true, fill: false },
            points: { show: true , radius: 10},
            color: "green"
        });

        d6 = []
        lat = data["segment_points"][data["segment_points"].length - 1]["lat"];
        lng = data["segment_points"][data["segment_points"].length - 1]["lng"];
        var xy = project_web_mercator(lng, lat);
        d6.push(xy);
        map_plotdata.push({
            data: d6,
            lines: { show: true, fill: false },
            points: { show: true , radius: 10},
            color: "red"
        });




        options = { legend: { show: false }, xaxis: { show: false }, yaxis: { show: false }, grid:{autoHighlight:false, hoverable:true, clickable:false, axisMargin:10, borderWidth:0}};

        $("#profilemap").html("");
        var  plot = $.plot("#profilemap", map_plotdata, options);
        var opts = plot.getOptions();
        console.log(opts);
        opts.xaxes[0].max = opts.xaxes[0].min + (opts.yaxes[0].max - opts.yaxes[0].min) * $('#profilemap').width() / $('#profilemap').height();
        plot.setupGrid();
        plot.draw();
  
        $("#powerform").show()
    })
        .fail(function () {
            $("#profilemap").html("Error getting profile");
            $("#profileresult").html("");
        });
}

var colormap = chroma.scale(['purple', 'blue', 'green', 'yellow', 'red', 'black']);
$(document).ready(function () {
    $("#powerform").hide();
    $("#windslider").roundSlider({
        min: 0,
        max: 359,
        value: 90,
        editableTooltip: false,
        showTooltip: false,
        change: function (e) {
            slider_value = e.value - 90;
            if (slider_value < 0) {
                slider_value += 360;
            }
        }
    });
    $("#profileform").submit(profile);
    $("#powerform").submit(function (event) {
        profile(event);
        event.preventDefault();
        var compare = $("#select_type").val() == "effort" &&  $("#check_compare").is(":checked");
        $("#powerresult").html('<div class="spinner-border m-5" role="status"> <span class="sr-only">Loading...</span> </div>');
        url = root + "/api/optimal_power" + "?mass=" + String($("#mass").val()) + "&power=" + String($("#power").val()) +
            "&id=" + String($("#segmentid").val()) + "&Cda=" + String($("#cda").val()) +
            "&Cr=" + String($("#cr").val()) +
            "&type=" + String($("#select_type").val()) +
            "&wp0=" + String($("#wp0").val()) +
            "&wind_speed=" + String($("#windspeed").val()) +
            "&wind_direction=" + String(slider_value) +
            "&temp=" + String($("#temp").val()) +
            "&use_ign=" + String(1.0 * ($("#correctalt").is(":checked"))) +
            "&optim_type=" + String($("#select_optim_type").val()) +
            "&use_gmaps=" + String(1.0 * ($("#snaptoroads").is(":checked"))) +
            "&pen=" + String($("#pen").val()) +
            "&drivetrain_efficiency=" + String($("#efficiency").val());
        $.getJSON(url, function (data) {
            var items = [];
            html = '<table class="table-striped"><thead class="thead-dark m-5"> <tr>' +
                '<th scope="col">Distance (km)</th>' +
                '<th scope="col">Length (km)</th> <th scope="col">Elevation (m)</th>' +
                '<th scope="col">Time</th>' +
                '<th scope="col">Grade (%)</th>' +
                '<th scope="col">Power (W)</th>' +
                '<th scope="col">Speed (km/h)</th>' +
                '<th scope="col">Average (km/h)</th>' +
                '<th scope="col">Headwind (km/h)</th>' +
                '</tr> </thead>';
            var idx = 0;
            $.each(data["intervals"], function (key, val) {
                var date = new Date(0);
                var t = val['time']
                date.setSeconds(t); // specify value for SECONDS here
                var timeString = date.toISOString().substr(11, 8);
                items.push('<tr><th scope="row">' + (Number.parseFloat(val['distance']) / 1000).toFixed(1) +
                    ' </th><td>' + (Number.parseFloat(val['length']) / 1000).toFixed(1) +
                    ' </td><td>' + Number.parseFloat(val['total_elevation']).toFixed(1) +
                    ' </td><td>' + timeString +
                    ' </td><td>' + Number.parseFloat(val['grade']).toFixed(1) +
                    ' </td><td>' + Number.parseFloat(val['est_power']).toFixed(1) +
                    ' </td><td>' + Number.parseFloat(val['speed_kmh']).toFixed(1) +
                    ' </td><td>' + Number.parseFloat(val['average_kmh']).toFixed(1) +
                    ' </td><td>' + (Number.parseFloat(val['headwind'])*3.6).toFixed(1) +
                    '</td></tr>');
                if (compare) {
                    var val = profile_data["profile_intervals"][idx];
                    var date_effort = new Date(0);
                    date_effort.setSeconds(val['time']); // specify value for SECONDS here
                    var dt = val['time'] - t;
                    if (dt > 0) {
                        dt = "+ " + dt.toFixed(0);
                    }
                    else {
                        dt = "- " + (-dt).toFixed(0);
                    }
                    var timeString_effort = date_effort.toISOString().substr(11, 8);
                    idx++;
                    items.push('<tr><th scope="row"> '+
                    ' </th><td>' + 
                    ' </td><td>Compared:' +
                    ' </td><td>' + timeString_effort + 
                    ' </td><td>' + '(' + dt + 's)' +
                    ' </td><td>' + Number.parseFloat(val['est_power']).toFixed(1) +
                    ' </td><td>' + Number.parseFloat(val['speed_kmh']).toFixed(1) +
                    ' </td><td>' + Number.parseFloat(val['average_kmh']).toFixed(1) +
                    ' </td><td>' +
                    '</td></tr>');
                }
            });
            $("#powerresult").html(

                $("<table/>", {
                    "class": "table",
                    html: html + items.join("") + "</table>"
                }))
        });
    });
});
