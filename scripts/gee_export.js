// Google Earth Engine export of the MODIS LST tiles used in the paper.
// Paste into the Earth Engine Code Editor (https://code.earthengine.google.com)
// and run. It starts one Drive export per tile and year (5 tiles x 11 years).
//
// Output: <tile>_<year>.tif, EPSG:4326, ~1 km, float32, bands named
//   YYYY_MM_DD_T_day, YYYY_MM_DD_T_night, YYYY_MM_DD_T_dayerr, YYYY_MM_DD_T_nighterr
//   (all Terra composites of the year), then the same four for Aqua (A_*).
// LST in deg C (DN*0.02 - 273.15). Kept where QC bits 0-1 are 0 or 1 ("LST produced").
// *_err = MODIS LST error class, QC bits 6-7 (0: <=1 K, 1: <=2 K, 2: <=3 K, 3: >3 K).

var TILES = {
  thar_arid:       ee.Geometry.Rectangle([71.1735, 26.9045, 71.8293, 27.4884]),
  deccan_semiarid: ee.Geometry.Rectangle([77.3000, 17.0051, 77.9019, 17.5890]),
  bandhavgarh:     ee.Geometry.Rectangle([80.7136, 23.4101, 81.3514, 23.9940]),
  ne_humid:        ee.Geometry.Rectangle([92.4816, 25.9074, 93.1283, 26.4913]),
  konkan_coastal:  ee.Geometry.Rectangle([72.9971, 16.6098, 73.6080, 17.1938])
};
var YEARS = ee.List.sequence(2015, 2025).getInfo();
var FOLDER = 'lst_dark_composites';

function prep(prefix) {
  return function (img) {
    var lst = function (band, qcBand) {
      var qc = img.select(qcBand);
      var ok = qc.bitwiseAnd(3).lte(1);                       // bits 0-1: 00 or 01
      var t = img.select(band).multiply(0.02).subtract(273.15).updateMask(ok);
      var err = qc.rightShift(6).bitwiseAnd(3).updateMask(ok); // bits 6-7
      return [t, err];
    };
    var d = lst('LST_Day_1km', 'QC_Day'), n = lst('LST_Night_1km', 'QC_Night');
    return ee.Image.cat([d[0], n[0], d[1], n[1]]).toFloat()
      .rename([prefix + '_day', prefix + '_night', prefix + '_dayerr', prefix + '_nighterr']);
  };
}

Object.keys(TILES).forEach(function (name) {
  var region = TILES[name];
  YEARS.forEach(function (y) {
    var start = ee.Date.fromYMD(y, 1, 1), end = start.advance(1, 'year');
    var terra = ee.ImageCollection('MODIS/061/MOD11A2').filterDate(start, end).map(prep('T')).toBands();
    var aqua  = ee.ImageCollection('MODIS/061/MYD11A2').filterDate(start, end).map(prep('A')).toBands();
    Export.image.toDrive({
      image: terra.addBands(aqua).clip(region),
      description: name + '_' + y, folder: FOLDER, fileNamePrefix: name + '_' + y,
      region: region, crs: 'EPSG:4326', scale: 1000, maxPixels: 1e9
    });
  });
});
