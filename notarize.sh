#!/bin/zsh
# https://lessons.livecode.com/m/4071/l/1122100-codesigning-and-notarizing-your-lc-standalone-for-distribution-outside-the-mac-appstore
application_id="Developer ID Application: Kevin Chevalier (UYEFTYB8JB)"
username="kevin.chevalier10@gmail.com"
password_id="TeachApp"
bundle_id="com.chevalier.teachapp.zip"

app_path="/Users/kevin/Documents/Repos/TeachApp/dist/TeachApp.app"
dmg_name="TeachApp.dmg"
dmg_path="/Users/kevin/Documents/Repos/TeachApp/dist/$dmg_name"
volume_name="TeachAppML"

got_requestUUID=""

if [ -z "$got_requestUUID" ];
then
    echo "Setting attributes and permissions"
    sudo xattr -cr "$app_path"
    sudo xattr -lr "$app_path"
    sudo chmod -R u+rw "$app_path"
    echo "\nSigning application"
    codesign --deep --force --verify --verbose --sign "$application_id" --options runtime "$app_path"
    # codesign -f -s "$application_id" --options runtime "$app_path" --deep
    # codesign --verify --verbose "$app_path"
    # echo "\nCreating DMG File"
    # hdiutil create -volname "$volume_name" -size 150m -srcfolder "$app_path" -ov -format UDZO "$dmg_path"
    # echo "\nSigning DMG File"
    # codesign -f -s "$application_id" --options runtime "$dmg_path" --deep
    # # codesign --deep --force --verify --verbose --sign "$application_id" --options runtime "$dmg_path"
    # codesign --verify --verbose "$dmg_path"
    # echo "\nUploading DMG File for notarization. (Store Request UUID)"
    # xcrun altool -type osx --notarize-app --primary-bundle-id "$bundle_id" --username "$username" --password "@keychain:$password_id" --file "$dmg_path"
else
    echo "\nChecking notarization status"
    xcrun altool --notarization-info "$got_requestUUID" --username "$username" --password "@keychain:$password_id"
fi
