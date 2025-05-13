import {
    changePage,
    getDetails,
    setDetails,
    createId,
} from "../details-functions.js";
import { createDropdown, validateShortcut, normalizeShortcut } from "../widgets/widgets-base.js";
import { createReorderColumns } from "./main-page.js";

function createDropdownPage(id, data) {
    d3.select(id).selectAll("*").remove();

    let mb = d3
        .select(id)
        .append("div")
        .attr("id", "dropdown-page-mb")
        .attr("class", "modal-body");

    // explanation text
    mb.append("h6").text("Create Dropdown Widget");

    // example
    mb.append("div")
        .attr("id", "dropdown-page-example")
        .attr("class", "center example");
    createDropdown(
        "#dropdown-page-example",
        data
            ? { ...data, id: "sample-dropdown" }
            : {
                  id: "sample-dropdown",
                  title: "Detail Name",
                  options: [
                      { value: "Option 1", shortcut: "1", selected: true },
                      { value: "Option 2", shortcut: "2" },
                  ],
              }
    );

    mb.append("div").text(
        "Enter the detail name and create options for the dropdown. There must be 2-5 options. Also select which option should be selected by default."
    );
    mb.append("hr");
    // title
    let form = mb
        .append("form")
        .attr("class", "need-validation")
        .attr("novalidate", "true");
    let nameDiv = form
        .append("div")
        .attr("class", "form-group position-relative");
    nameDiv
        .append("label")
        .attr("for", "dropdown-title")
        .attr("class", "form-label")
        .text("Detail Name");
    nameDiv
        .append("input")
        .attr("type", "text")
        .attr("class", "form-control")
        .attr("id", "dropdown-title")
        .property("value", data ? data.title : "")
        .property("readonly", data && data.isDefault)
        .property("disabled", data && data.isDefault)
        .on("keydown", function(e) {
            if (data && data.isDefault) {
                e.preventDefault();
                return false;
            }
        });
    nameDiv
        .append("div")
        .attr("class", "invalid-tooltip")
        .text(
            "Detail names must be 1-16 characters long, and can only contain alphanumeric characters, dashes, underscores, and spaces."
        );

    // options
    let optionsDiv = form
        .append("div")
        .attr("class", "form-group position-relative")
        .attr("id", "options-div");
    optionsDiv
        .append("label")
        .attr("for", "dropdown-options")
        .attr("class", "form-label")
        .text("Options");
    optionsDiv.append("div").attr("id", "dropdown-options");

    const defaultOptions = [
        { value: "Option 1", shortcut: "1" },
        { value: "Option 2", shortcut: "2", selected: true },
    ];
    const options = data ? data.options.map(opt => ({
        value: opt.value,
        shortcut: opt.shortcut || '',
        selected: opt.selected || false
    })) : defaultOptions;
    options.forEach(option => createOption(option, data && data.isDefault));
    createAddOptionButton();
    optionsDiv
        .append("div")
        .attr("class", "invalid-tooltip")
        .text("Options must be 1-32 characters long and unique.");

    // footer
    let footer = d3.select(id).append("div").attr("class", "footer-row");
    footer
        .append("button")
        .attr("type", "button")
        .attr("class", "grey-btn")
        .text("Back")
        .on(
            "click",
            data
                ? () => changePage(id, "#main-page")
                : () => changePage(id, "#widget-type-page")
        );

    footer
        .append("button")
        .attr("type", "button")
        .attr("class", "grey-btn")
        .text("Create Dropdown")
        .on(
            "click",
            data ? () => createNewDropdown(data) : () => createNewDropdown()
        );
}

function createOption(option, number) {
    number += 1;
    let div = d3
        .select("#dropdown-options")
        .append("div")
        .attr("class", "form-check new-option")
        .attr("id", `dropdown-option-${number}`);

    div.append("input")
        .attr("class", "form-check-input")
        .attr("type", "radio")
        .attr("name", "dropdown-options")
        .attr("id", `new-dropdown-${number}`)
        .attr("value", `dropdown-option-${number}`)
        .attr("checked", option.selected);

    // Container für Option und Shortcut
    let inputContainer = div.append("div")
        .attr("class", "d-flex align-items-center gap-2");

    // Option Input
    inputContainer.append("input")
        .attr("type", "text")
        .attr("class", "form-control")
        .attr("placeholder", "Option")
        .attr("value", option.value)
        .attr("data-original-value", option.value);

    // Shortcut Input
    inputContainer.append("input")
        .attr("type", "text")
        .attr("class", "form-control")
        .attr("placeholder", "Shortcut")
        .attr("value", option.shortcut || "")
        .attr("data-original-shortcut", option.shortcut || "");

    if (number > 2) {
        div.append("i")
            .attr("class", "bi bi-trash-fill")
            .on("click", () => {
                d3.select(`#dropdown-option-${number}`).remove();
                if (getNumOptions() === 4) {
                    createAddOptionButton();
                }
            });
    }
}

function createAddOptionButton(id = "#dropdown-page") {
    d3.select(id)
        .select("#options-div")
        .append("button")
        .text("Add Option")
        .attr("class", "grey-btn add-option-btn")
        .on("click", function(e) {
            e.preventDefault();
            let number = getNumOptions();
            createOption({ value: `Option ${number + 1}` }, number);
            if (number >= 5) {
                d3.select(this).remove();
            }
        });
}

function getNumOptions(id = "#dropdown-page") {
    return d3
        .select(id)
        .selectAll(".new-option")
        .size();
}

function createNewDropdown(data) {
    // input sanitization
    let invalid = false;

    const title = d3.select("#dropdown-title").property("value");
    if (
        title.length < 1 ||
        title.length > 16 ||
        !/^[_a-zA-Z0-9- ]*$/.test(title)
    ) {
        d3.select("#dropdown-title").classed("is-invalid", true);
        invalid = true;
    } else {
        d3.select("#dropdown-title").classed("is-invalid", false);
    }

    // Sammle alle Optionen und Shortcuts
    let options = [];
    const selected = d3
        .select(`input[name="dropdown-options"]:checked`)
        .property("value");

    // Validiere und sammle Optionen
    d3.select("#dropdown-options")
        .selectAll(".new-option")
        .each(function() {
            let option = {};
            let inputs = d3.select(this).selectAll("input[type='text']");
            const valueInput = inputs.nodes()[0];
            const shortcutInput = inputs.nodes()[1];
            
            option.value = valueInput.value.trim();
            const shortcut = shortcutInput.value.trim();
            
            // Setze den Shortcut direkt
            if (shortcut) {
                option.shortcut = shortcut;
            }

            if (selected === d3.select(this).attr("id")) {
                option.selected = true;
            }
            options.push(option);
        });

    let optionValues = options.map(x => x.value);
    if (
        optionValues.some(value => value.length < 1 || value.length > 32) ||
        !_.isEqual(optionValues, _.uniq(optionValues))
    ) {
        d3.select("#dropdown-options").classed("is-invalid", true);
        invalid = true;
    } else {
        d3.select("#dropdown-options").classed("is-invalid", false);
    }

    if (invalid) {
        return;
    }

    // actual creation
    let details = getDetails();
    const newDetail = {
        type: "dropdown",
        title: title,
        id: createId(title),
        options: options,
        editable: true,
    };
    if (data) {
        let i = _.findIndex(details, data);
        details.splice(i, 1, newDetail);
    } else {
        details.push(newDetail);
    }
    setDetails(details);
    createReorderColumns("#reorder");

    changePage("#dropdown-page", "#main-page");
}

export { createDropdownPage };
