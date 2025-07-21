$(document).ready(function () {
    var stop = false;

    function fetchData() {
        $.getJSON('/data', function (data) {
            var inputText = '';
            var outputText = '';
            for (var i = data.input_data.length - 1; i >= 0; i--) { // Iterate from bottom to top
                inputText += data.input_data[i].join(', ') + '\n';
                var attackType = data.predicted_data[i];
                // Define colors for each attack type here
                var colorClass = getColorClassForAttackType(attackType);
                outputText += '<span class="' + colorClass + '">' + attackType + '</span>\n';
            }
            $('#input-data').text(inputText);
            $('#output-data').html(outputText); // Use .html() to render HTML with color classes

            if (!stop) {
                setTimeout(fetchData, 1000); // Fetch data every second
            }
        });
    }

    // Function to map attack types to color classes
    function getColorClassForAttackType(attackType) {
        switch (attackType) {
            case 'Analysis':
                return 'attack-analysis';
            case 'DoS':
                return 'attack-dos';
            case 'Exploits':
                return 'attack-exploits';
            case 'Normal':
                return 'attack-normal';
            default:
                return 'attack-default'; // Default color if type not defined
        }
    }

    $('#stop-form').submit(function (event) {
        event.preventDefault();
        stop = true;
        $.ajax({
            type: 'POST',
            url: '/stop',
            success: function () {
                console.log('Prediction stopped.');
            },
            error: function (error) {
                console.error('Error stopping prediction:', error);
            }
        });
    });

    fetchData();
});