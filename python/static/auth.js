$(document).ready(function () {
    $.ajax({
        url: root + "/api/auth"
    }).then(function (data) {
        if (data.auth == false) {
            // window.location.replace(root);
            $('#auth').html('<a class="navbar-brand" href="auth"><img src="strava.png" alt=""></a>');
            $('#main').html("");
           
        } else {
            //$('#name').append("Hello, ", data.firstname)
        }
    });
});