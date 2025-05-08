import { cfgDetails } from "../config-details.js";

function createRadioButtons(selectId, { id, title, options }) {
    d3.select(selectId)
        .append("div")
        .attr("class", cfgDetails.detailClass)
        .attr("id", id)
        .append("h3")
        .text(title)
        .attr("class", "center");

    for (let option of options) {
        let div = d3
            .select("#" + id)
            .append("div")
            .attr("class", "form-check vertical");

        div.append("input")
            .attr("class", "form-check-input")
            .attr("type", "radio")
            .attr("name", id)
            .attr("id", option.value) // sanitize, make sure no duplicate values
            .attr("value", option.value)
            .attr("checked", option.checked);
        div.append("label")
            .attr("class", "form-check-label")
            .attr("for", option.value)
            .text(option.value);
    }
}

function createTextField(
    selectId,
    { id, title, defaultValue, dataTableEditable }
) {
    let div = d3
        .select(selectId)
        .append("div")
        .attr("class", cfgDetails.detailClass + " " + "even-width")
        .attr("id", id);
    div.append("h3").text(title).attr("class", "center");
    div.append("div")
        .attr("class", "form-group")
        .append("input")
        .attr("type", "text")
        .attr("class", "form-control")
        .attr("value", defaultValue);
}

function createDropdown(selectId, { id, title, options }) {
    let div = d3
        .select(selectId)
        .append("div")
        .attr("class", cfgDetails.detailClass + " even-width")
        .attr("id", id);

    div.append("h3").text(title).attr("class", "center");

    let select = div
        .append("div")
        .append("select")
        .attr("id", id + "-select")
        .attr("class", "select2");

    options.forEach((option) => {
        const optionText = option.shortcut
            ? `${option.value} (${option.shortcut})`
            : option.value;

        select
            .append("option")
            .text(optionText)
            .attr("value", option.value)
            .attr("selected", option.selected);

        if (option.shortcut) {
            shortcutToDropdown.set(option.shortcut, {
                selectId: `${id}-select`,
                value: option.value,
            });
        }
    });
}

// Event-Listener für Tastaturbefehle
document.addEventListener('keydown', function(event) {
    options.forEach((option) => {
        // Überprüfen, ob der Shortcut gedrückt wurde
        if (event.key === option.shortcut) { // Verwende event.key
            const dropdownId = `#${id}-select`;
            d3.select(dropdownId).property('value', option.value); // Setze den Wert basierend auf dem Shortcut
            console.log(`Setting dropdown value to: ${option.value}`); // Log für Debugging
            
        }
    });
});


function createTimeWidget(selectId, { id, title, defaultTime, countdown }) {
    let div = d3
        .select(selectId)
        .append("div")
        .attr("class", cfgDetails.detailClass + " even-width")
        .attr("id", id);
    div.append("h3").text(title).attr("class", "center");

    // Standardwert für defaultTime mit Hundertstel
    if (defaultTime && !defaultTime.includes('.')) {
        defaultTime = defaultTime + '.00';
    } else if (!defaultTime) {
        defaultTime = '00:00.00';
    }
    
    // Füge ein Eingabefeld hinzu, um die Zeit anzuzeigen
    div.append("input")
        .attr("type", "text")
        .attr("readonly", true) // Nur Lesezugriff
        .attr("value", "00:00.00"); // Standardwert mit Hundertstel

    const videoElement = document.getElementById('gameVideo'); // Das Video-Element

    // Event-Listener für die Zeitaktualisierung
    videoElement.addEventListener('timeupdate', function() {
        const currentTime = videoElement.currentTime; // Aktueller Zeitstempel des Videos
        const formattedTime = formatTime(currentTime); // Zeit formatieren
        d3.select("#" + id).select("input").property("value", formattedTime); // Aktualisiere das Widget
    });

    // Hilfsfunktion zum Formatieren der Zeit mit Hundertstel
    function formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        const hundredths = Math.floor((seconds % 1) * 100);
        return `${minutes}:${secs < 10 ? '0' : ''}${secs}.${hundredths < 10 ? '0' : ''}${hundredths}`; // Format: mm:ss.hh
    }
}



const shortcutToDropdown = new Map();

document.addEventListener("keydown", (event) => {
    for (const [shortcut, { selectId, value }] of shortcutToDropdown.entries()) {
        if (event.key === shortcut) {
            d3.select(`#${selectId}`).property("value", value).dispatch("change");
        }
    }
});


export {
    createRadioButtons,
    createTextField,
    createDropdown,
    createTimeWidget,
    shortcutToDropdown,
};

