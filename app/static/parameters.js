root = ""



$(document).ready(function () {
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
    $("#estimation_form").submit(function (event) {
        event.preventDefault();
        $("#powerresult").html('<div class="spinner-border m-5" role="status"> <span class="sr-only">Loading...</span> </div>');
        url = root + "/api/parameters" +
            "?id=" + String($("#effortid").val()) +
            "&type=effort" +
            "&key=power" +
            "&wind_speed=" + String($("#windspeed").val()) +
            "&wind_direction=" + String(slider_value) +
            "&temp=" + String($("#temp").val()) +
            "&pen=" + String($("#pen").val());
        if($("#estimate_mass").is(":checked")) {
            url += "&mass=0";
        } else {
            url += "&mass=" + String($("#mass").val());
        }
        if($("#estimate_cda").is(":checked")) {
            url += "&Cda=0";
        } else {
            url += "&Cda=" + String($("#cda").val());
        }
        if($("#estimate_cr").is(":checked")) {
            url += "&Cr=0";
        } else {
            url += "&Cr=" + String($("#cr").val());
        }
        if($("#estimate_dt").is(":checked")) {
            url += "&drivetrain_efficiency=0";
        } else {
            url += "&drivetrain_efficiency=" + String($("#drivetrain_efficiency").val());
        }
        console.log(url);
        $.getJSON(url, function (data) {
            $("#mass").val(data["mass"]);
            $("#cda").val(data["Cda"]);
            $("#cr").val(data["Cr"]);
            $("#drivetrain_efficiency").val(data["drivetrain_efficiency"]);
        });
        $("#powerresult").html('Done.');
    });
});
