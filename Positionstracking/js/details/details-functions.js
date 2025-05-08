import { dataStorage } from "../../setup.js";

function getDetails() {
    return getCustomSetup().details;
}

function setDetails(detailsList) {
    setCustomSetup({ ...getCustomSetup(), details: detailsList });
}

export function getCustomSetup() {
    return dataStorage.get("customSetup");
}

export function setCustomSetup(setup) {
    dataStorage.set("customSetup", setup);
}

function existsDetail(id) {
    return !d3.select(id).empty();
}

export function getDetailTitle(id) {
    return _.find(getDetails(), ["id", _.trim(id, "#")]).title;
}

export function setCustomSetupUploadFlag(bool) {
    dataStorage.set("customSetupUploadFlag", bool);
}

export function resetCustomSetupUploadFlag() {
    let value = dataStorage.get("customSetupUploadFlag");
    setCustomSetupUploadFlag(false);
    return value;
}

function getCurrentShotTypes() {
    let options = [];
    if (existsDetail("#shot-type")) {
        d3.select("#shot-type-select")
            .selectAll("option")
            .each(function () {
                let obj = {
                    value: d3.select(this).property("value"),
                };
                if (
                    d3.select("#shot-type-select").property("value") ===
                    obj.value
                ) {
                    obj["selected"] = true;
                }

                options.push(obj);
            });
    }
    return options;
}

function getTypeIndex(type) {
    if (!existsDetail("#shot-type")) {
        return 0;
    }
    return type ? _.findIndex(getCurrentShotTypes(), { value: type }) : 0;
}

function changePage(currentPageId, newPageId) {
    d3.select(currentPageId).attr("hidden", true);
    d3.select(newPageId).attr("hidden", null);
}

function createId(title) {
    // lowercase and replace all whitespace
    // if starts with a number, insert a dummy letter "a" at start
    let id = title
        .toLowerCase()
        .replace(/\s/g, "-") // lowercase and replace all whitespace
        .replace(/^\d/, (d) => "a" + d);

    while (
        _.findIndex(getDetails(), { id: id }) !== -1 ||
        id === "x2" ||
        id === "y2"
    ) {
        id += "0";
    }
    return id;
}

function saveCurrentSetup() {
    // based on select2, reorder and tag with hidden
    const details = getDetails("details");
    let newDetails = [];
    d3.select("#reorder-columns")
        .selectAll("td")
        .each(function () {
            let detail = _.find(details, {
                id: d3.select(this).attr("data-id"),
            });
            if (
                d3.select(this).select("i").size() !== 0 &&
                d3.select(this).select("i").attr("class") ===
                    "bi bi-eye-slash-fill"
            ) {
                detail["hidden"] = true;
            } else {
                detail["hidden"] = null;
            }
            // custom saves for each
            if (!detail.hidden && detail.id) {
                const d = d3.select("#details").select("#" + detail.id);
                if (!d.empty()) {
                    switch (detail.type) {
                        case "team":
                            // save teams
                            detail.blueTeamName = d
                                .select("#blue-team-name")
                                .property("value");
                            detail.orangeTeamName = d
                                .select("#orange-team-name")
                                .property("value");
                            detail.checked = d3
                                .select("input[name='team-bool']:checked")
                                .property("id");
                            break;

                        case "player":
                        case "text-field":
                            // save current entry
                            detail["defaultValue"] = d
                                .select("input")
                                .property("value");
                            break;
                        case "shot-type":
                            detail.options = getCurrentShotTypes();
                            break;
                        case "dropdown":
                            // save currently selected option
                            let selectedValue = d
                                .select("select")
                                .property("value");
                            detail.options = detail.options.map(function (o) {
                                let option = { 
                                    value: o.value,
                                    shortcut: o.shortcut || '',
                                    selected: o.value === selectedValue
                                };
                                return option;
                            });
                            // Preserve editable and isDefault properties for dropdowns
                            detail.editable = true;
                            detail.isDefault = detail.isDefault || false;  // Preserve isDefault property
                            break;
                        case "radio":
                            // save current selection
                            let checkedValue = d
                                .select(`input[name='${detail.id}']:checked`)
                                .property("value");
                            detail.options = detail.options.map(function (o) {
                                let option = { value: o.value };
                                if (o.value === checkedValue) {
                                    option.checked = true;
                                }
                                return option;
                            });
                            break;
                        case "time":
                            // save current time
                            detail["defaultTime"] = d
                                .select("input")
                                .property("value");
                            break;
                    }
                }
            }
            newDetails.push(detail);
        });

    const customSetup = {
        details: newDetails,
        rowsPerPage: d3.select("#page-size-field").property("value"),
        widgetsPerRow: d3.select("#widgets-per-row-dropdown").property("value"),
        heatMapEnable: d3.select("#heat-map-enable").property("checked"),
        twoPointEnable: d3.select("#two-point-enable").property("checked"),
    };
    setCustomSetup(customSetup);
}

function convertDefaultToEditable(detail) {
    if (detail.type === "dropdown") {
        return {
            ...detail,
            editable: true,
            isDefault: true,  // Mark JSON dropdowns as default
            options: detail.options.map(option => ({
                ...option,
                shortcut: option.shortcut || '',
                selected: option.selected || false
            }))
        };
    }
    return {
        ...detail,
        isDefault: false  // Mark non-dropdowns as non-default
    };
}

function initializeEditableDetails() {
    const defaultSetup = getDefaultSetup();
    // Set isDefault for dropdowns before converting to editable
    defaultSetup.details.forEach(detail => {
        if (detail.type === "dropdown") {
            detail.isDefault = true;
        }
    });
    const defaultDetails = defaultSetup.details.map(convertDefaultToEditable);
    setDetails(defaultDetails);
    createReorderColumns("#reorder");
}

export {
    getDetails,
    setDetails,
    existsDetail,
    getCurrentShotTypes,
    getTypeIndex,
    changePage,
    createId,
    saveCurrentSetup,
    convertDefaultToEditable,
    initializeEditableDetails
};
