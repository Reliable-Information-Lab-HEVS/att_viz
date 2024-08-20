/**
 * @fileoverview Transformer Visualization D3 javascript code.
 *
 *
 *  Based on: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/visualization/attention.js and https://github.com/jessevig/bertviz/blob/master/bertviz/head_view.js
 *
 * Change log:
 *
 * 12/19/18  Jesse Vig     Assorted cleanup. Changed orientation of attention matrices.
 * 12/29/20  Jesse Vig     Significant refactor.
 * 12/31/20  Jesse Vig     Support multiple visualizations in single notebook.
 * 02/06/21  Jesse Vig     Move require config from separate jupyter notebook step
 * 05/03/21  Jesse Vig     Adjust height of visualization dynamically
 * 07/25/21  Jesse Vig     Support layer filtering
 * 03/23/22  Daniel SC     Update requirement URLs for d3 and jQuery (source of bug not allowing end result to be displayed on browsers)
 * 05/07/24  Ana Indreias  Significant refactor: self-attention visualization. None other compatible.
 **/

require.config({
    paths: {
        d3: 'https://cdnjs.cloudflare.com/ajax/libs/d3/5.7.0/d3.min',
      jquery: 'https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.0/jquery.min',
    }
});
  
requirejs(['jquery', 'd3'], function ($, d3) {
      
    const params = PYTHON_PARAMS; // HACK: this is a template marker that will be replaced by the actual params.

    /**
     * The font size. Also used for setting (dx, dy) for text placement.
     * @constant {number}
     */
    const TEXT_SIZE = 15;

    /**
     * The height of a token box.
     * @constant {number}
     */
    const BOXHEIGHT = 22.5;

    /**
     * The space between the two attention rendition modes.
     * @constant {number}
     */
    const MATRIX_WIDTH = 115;

    /**
     * The width of a self-attention head selection box.
     * @constant {number}
     */
    const CHECKBOX_SIZE = 20;

    /**
     * Where the HTML render should begin (on the y-axis)
     * @constant {number}
     */
    const TEXT_TOP = 30;

    /**
     * The maximum length of a line of text.
     * @constant {number}
     */
    const LINE_WIDTH = 870;

    console.log("d3 version", d3.version)
    
    /**
     * Colour scheme used to represent the self-attention heads.
     * 
     * See https://d3js.org/d3-scale-chromatic/categorical#schemeCategory10.
     */
    let headColours;
    try {
        headColours = d3.scaleOrdinal(d3.schemeCategory10); // 
    } catch (err) {
        console.log('Older d3 version')
        headColours = d3.scale.category10();
    }

    /**
     * Global variable for passing configuration information such as the attention matrix,
     * the completion tokens etc.
     */
    let config = {};

    /**
     * Global boolean representing whether a token in the first view ("Observed" view) was clicked.
     */
    let clickObservedView = false;

    /**
     * Global boolean representing whether a token in the second view ("Observer" view) was clicked.
     */
    let clickObserverView = false;

    initializeConfig();

    renderVisualization();

    /**
     * Initializes the global variable config, as well as the HTML file.
     * 
     * Installs change listeners to re-render the visualization when needed.
     */ 
    function initializeConfig() {
        config.attention = params['attention'];
        config.rootDivId = params['root_div_id'];
        config.nLayers = config.attention['num_layers'];
        config.nHeads = config.attention['num_heads'];
        config.layers = [...Array(config.nLayers).keys()]; // equivalent to range(nLayers); see https://stackoverflow.com/a/10050831

        config.totalDy = config.attention['dy_total'];
        config.tokenInfo = config.attention['pos']

        config.layerSeq = 0

        config.layer = config.layers[config.layerSeq]
        config.head = 0
        config.headStartIdx = config.attention['head_start_idx']

        // Mark the first head as selected / the default view
        config.headVis = new Array(config.nHeads).fill(false);
        config.headVis[config.head] = true;

        // Build the layer selector + install change listener
        if (config.nLayers > 1) {
            let layerEl = $(`#${config.rootDivId} #layer`);
            for (const layer of config.layers) {
                layerEl.append($("<option />").val(layer).text(layer));
            }
            layerEl.val(config.layer).change();

            // When the current layer changes, re-render the
            // visualization (using the new attention values):
            layerEl.on('change', function (e) {
                config.layer = +e.currentTarget.value;
                config.layerSeq = config.layers.findIndex(layer => config.layer === layer);
                renderVisualization();
            });
        }
    }

    /**
     * Renders the HTML visualization.
     * 
     * There are two views:
     * 
     * **View 1 (top): The 'Observed' view**
     * 
     * Hover over a token to see which successor tokens
     * have high attention values towards it.
     * 
     * **View 2 (bottom): The 'Observer' view**
     * 
     * Hover over a token to see which previous tokens
     * have impacted its generation through self-attention.
     * 
     * For either view, you can double-click a token to fix
     * the self-attention visualization, and double-click again to undo this.
     */
    function renderVisualization() {
        clickObservedView = false;
        clickObserverView = false;

        // Load parameters
        const attnData = config.attention;
        const tokens = attnData.tokens;
        const promptLength = attnData.prompt_length; // The prompt length in tokens

        // Select attention for given layer
        const layerAttention = Array.from(attnData.attn[config.layer])

        // Clear visualization
        $(`#${config.rootDivId} #vis`).empty();

        // Determine size of the new visualization (TODO this is always the same, can save it as a constant somewhere)
        const height = MATRIX_WIDTH + 2*(config.totalDy + BOXHEIGHT) + TEXT_TOP;
        const width = LINE_WIDTH + 20;
        const svg = d3.select(`#${config.rootDivId} #vis`)
            .append('svg')
            .attr("width", width + "px")
            .attr("height", height + "px");

        // tokenInfo contains (dx, dy, width) values for each token (the height is constant: see `BOXHEIGHT`)
        renderText(svg, tokens, true,  layerAttention, promptLength, config.tokenInfo); // Observed view
        renderText(svg, tokens, false, layerAttention, promptLength, config.tokenInfo.map(x => [x[0], x[1] + MATRIX_WIDTH + config.totalDy, x[2], x[3]])); // Observer view

        if (config.nHeads > 1)
            drawCheckboxes(0, svg, config.headStartIdx);

        svg.select("#attention").attr("visibility", "hidden");
    }

    /**
     * Renders the given text and attention in the svg object.
     * 
     * @param {*} svg the svg object in which to render the tokens and attention
     * @param {*} text the prompt + completion tokens to render
     * @param {boolean} isObserved whether the view focuses on observed (prompt) tokens, or observer (completion) tokens
     * @param {*} attention the self-attention matrix
     * @param {number} promptLength the length of the prompt (in tokens)
     * @param {*} tokenInfo contains (dx, dy, width) values for each token (the height is constant: see `BOXHEIGHT`)
     */
    function renderText(svg, text, isObserved, attention, promptLength, tokenInfo) {
        const textContainer = svg.append("svg:g")
            .attr("id", isObserved ? "observed" : "observer");

        const tokenContainer = textContainer.append("g").selectAll("g")
            .data(text)
            .enter()
            .append("g");

        // Add gray background that appears when hovering over text
        tokenContainer.append("rect")
            .classed("background", true)
            .style("opacity", 0.0)
            .attr("fill", "lightgray")
            .attr("y", (_, i) => tokenInfo[i][1])
            .attr("width", (_, i) => tokenInfo[i][2])
            .attr("height", BOXHEIGHT);

        let textWidth = new Array(tokenInfo.length).fill(0);

        // Add token text
        const textEl = tokenContainer.append("text")
            .text(d => d)
            .attr("font-size", TEXT_SIZE + "px")
            .attr("font-weight", (_, i) => ((i < promptLength) ? "bold" : "normal"))
            .style("cursor", "default")
            .style("-webkit-user-select", "none")
            .attr("y", (_, i) => tokenInfo[i][1])
            .each(function(_, i) {
                textWidth[i] = 0.3*(0.25 + tokenInfo[i][3])*TEXT_SIZE + this.getComputedTextLength();
                if (i != 0 && tokenInfo[i-1][1] == tokenInfo[i][1]) {
                    tokenInfo[i][0] = tokenInfo[i-1][0] + textWidth[i-1];
                }
            });

        textEl.style("text-anchor", "start")
            .attr("dx", (_, i) => +0.3*(0.25+tokenInfo[i][3])*TEXT_SIZE)
            .attr("dy", TEXT_SIZE);

        textContainer.selectAll("rect")
            .attr("width", (_, i) => textWidth[i])
            .attr("x", (_, i) => tokenInfo[i][0]);

        textContainer.selectAll("text")
            .attr("x", (_, i) => tokenInfo[i][0]);

        // Mouse Over (hover) listener: show gray background for moused-over token
        // if and only if the view has not been double clicked. + TODO: document the coloring functionality
        tokenContainer.on("mouseover", function (_, index) {
            if (!(isObserved ? clickObservedView : clickObserverView)) {
                textContainer.selectAll(".background")
                    .style("opacity", function (_, i) {
                        if (i === index) {
                            return 1.0
                        } else if (isObserved) {
                            if (i >= promptLength && index <= i) { // Can only be observed by i if : index <= i >= prompt length
                                return attention[config.head][i - promptLength][index];
                            }
                        } else if (index >= promptLength) { // is an observer => can only observe if index >= prompt length
                            if (i <= index) // Highlight only items with smaller indices than index
                                return attention[config.head][index - promptLength][i];
                        }

                        return 0;
                    })
                    .attr("fill", function (_, i) {
                        return i === index ? "lightgray" : headColours(config.head)
                    })
            } 
        });

        // Double Click listener: toggle the global booleans
        textContainer.on("dblclick", function () {
            if (isObserved)
                clickObservedView = !clickObservedView;
            else 
                clickObserverView = !clickObserverView;
        });

        // Mouse Leave listener: remove visualizations if
        // and only if the view has not been double-clicked
        textContainer.on("mouseleave", function () {
            if (!(isObserved ? clickObservedView : clickObserverView)) {
                // Unhighlight selected token
                d3.select(this).selectAll(".background")
                    .style("opacity", 0.0);
            }
        });
    }

    /**
     * Renders the attention-head selection boxes.
     * 
     * @param {*} top the y-coordinate where the boxes should be rendered
     * @param {*} svg the svg object in which to render the selection boxes
     * @param {*} head_start_idx the starting head index (useful for chunking)
     */
    function drawCheckboxes(top, svg, head_start_idx) {
        const checkboxContainer = svg.append("g");

        const headContainer = checkboxContainer.append("g").selectAll("g")
            .data(config.headVis)
            .enter()
            .append("g");

        const tmp = headContainer.append("rect")
            .attr("x", (_, i) => i * CHECKBOX_SIZE)
            .attr("y", top)
            .attr("width", CHECKBOX_SIZE)
            .attr("height", CHECKBOX_SIZE)
            .attr("fill", (_, i) => headColours(i));
            
        const textEl = headContainer.append("text")
            .text((_, i) => i + head_start_idx)
            .attr("font-size", 0.8*TEXT_SIZE + "px")
            .style("cursor", "default")
            .style("-webkit-user-select", "none")
            .attr("x", (_, i) => i * CHECKBOX_SIZE)
            .attr("y", top)
            .attr("width", CHECKBOX_SIZE)
            .attr("height", CHECKBOX_SIZE)
            .style("text-anchor", "start")
            .attr("dx", +0.1*TEXT_SIZE)
            .attr("dy", TEXT_SIZE);

        /**
         * Updates the colours of the self-attention head selection boxes based on
         * whether they are selected (default colour) or not (lighter colour).
         */
        function updateCheckboxes() {
            checkboxContainer.selectAll("rect")
                .attr("fill", (_, i) => config.headVis[i] ? headColours(i): lighten(headColours(i)));
        }

        updateCheckboxes();

        // The code below restricts self-attention head selection.
        
        // One (and only one) self-attention head must be selected at all times.

        // This means that:
        // - double clicks on the already-selected self-attention head will be ignored
        // - double clicks on a different self-attention head will unselect the current head,
        //   then select the chosen one
        textEl.on("dblclick", function (_, i) {
            if (!config.headVis[i] && activeHeads() === 1) {
                config.headVis = new Array(config.nHeads).fill(false);
                config.headVis[i] = true;
                config.head = i;

                // Since we're changing the selected head,
                // click status on the tokens needs to be reset as well.
                clickObservedView = false;
                clickObserverView = false;
            }
            updateCheckboxes();
        });
}

    /**
     * Returns a lighter version of the given colour.
     * 
     * @param {*} colour colour to lighten
     * @returns a lighter version of this colour
     */
    function lighten(colour) {
        const c = d3.hsl(colour);
        const increment = (1 - c.l) * 0.6;
        c.l += increment;
        c.s -= increment;
        return c;
    }

    /**
     * Returns the number of active heads.
     * 
     * @returns the number of active heads
     */
    function activeHeads() {
        return config.headVis.reduce(function (acc, val) {
            return val ? acc + 1 : acc;
        }, 0);
    }
});