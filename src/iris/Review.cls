Class iris.Review Extends %Persistent
{

Property ReviewLabel As %String;

Property ReviewText As %String(MAXLEN = 300000);

Storage Default
{
<Data name="ReviewDefaultData">
<Value name="1">
<Value>%%CLASSNAME</Value>
</Value>
<Value name="2">
<Value>ReviewLabel</Value>
</Value>
<Value name="3">
<Value>ReviewText</Value>
</Value>
</Data>
<DataLocation>^iris.ReviewD</DataLocation>
<DefaultData>ReviewDefaultData</DefaultData>
<ExtentSize>700000</ExtentSize>
<IdLocation>^iris.ReviewD</IdLocation>
<IndexLocation>^iris.ReviewI</IndexLocation>
<StreamLocation>^iris.ReviewS</StreamLocation>
<Type>%Storage.Persistent</Type>
}

}
